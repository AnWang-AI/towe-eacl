import torch


class LinearCRF(torch.nn.Module):

    def __init__(self, num_labels):
        torch.nn.Module.__init__(self)

        self.transitions = torch.nn.Parameter(torch.FloatTensor(num_labels, num_labels))
        torch.nn.init.xavier_uniform_(self.transitions, gain=0.2)
        # self.transitions = torch.nn.Parameter(torch.Tensor([[0, 0, 0, 0], [0, 0, 0, 1e10], [0, 0, 0, -1e10], [0, 0, 1e10, 0]]),
        #                                       requires_grad=True)

    def score_sentence(self, features, tags, mask=None):
        """计算给定目标标注序列的分数（分子项）
        mask (batch_size, seq_len): the non-zero positions will be ignored
        """
        # 计算emission scores
        # [batch_size, max_seq_len]
        emission_scores = torch.gather(features, dim=2, index=tags.unsqueeze(2)).squeeze(2)
        if mask is not None:
            emission_scores *= (1. - mask.to(features))
        emission_scores = torch.sum(emission_scores, dim=1)

        # 计算transition scores
        # 标签可转移数量等于序列长度减一, transitions[i, j]表示从标签i -> j的概率
        transition_scores = self.transitions[tags[:, :-1], tags[:, 1:]]
        if mask is not None:
            transition_scores *= (1. - mask[:, 1:].to(features))
        transition_scores = torch.sum(transition_scores, dim=1)

        scores = emission_scores + transition_scores
        return scores

    def forward_alg(self, features):
        """计算所有标注序列分数和（归一化分母项）"""
        # [seq_len, batch_size, num_tags]
        # 计算initial state
        forward_var = features[:, 0, :]
        for i in range(1, features.shape[1]):
            feature = features[:, i, :]
            # [batch_size, num_tags, num_tags]
            scores = forward_var.unsqueeze(2) + self.transitions + feature.unsqueeze(1)
            forward_var = torch.logsumexp(scores, dim=1)

        # [batch_size]
        log_norm = torch.logsumexp(forward_var, dim=1)

        return log_norm

    def neg_log_likelihood(self, input, target, mask=None, reduction="mean", weight=None):
        """Negative log likelihood as CRF loss"""
        if weight is not None:
            input *= weight

        gold_score = self.score_sentence(features=input, tags=target, mask=mask)
        # print(111111, gold_score, gold_score.shape)
        forward_score = self.forward_alg(features=input)
        # print(222222, forward_score, forward_score.shape)
        neg_log_likelihood = forward_score - gold_score

        if reduction == "mean":
            return neg_log_likelihood.mean()

        return neg_log_likelihood

    @torch.no_grad()
    def _viterbi_decode(self, features):
        """Decoding sequence tagging by viterbi algorithm"""
        # 动态规划之前向算法求解最优分数
        features = features.permute(1, 0, 2)
        forward_var = features[0]

        backpointers = []
        for feature in features[1:]:
            # [batch_size, num_tags, num_tags]
            forward_scores = forward_var.unsqueeze(2) + self.transitions
            # [batch_size, 1, num_tags]
            max_indices = forward_scores.argmax(1, keepdim=True)
            # [batch_size, num_tags]
            max_scores = forward_scores.gather(1, max_indices).squeeze(1)
            forward_var = feature + max_scores

            backpointers.append(max_indices.squeeze(1))

        # 后向求解最优路径
        # [batch_size]
        backward_indices = forward_var.argmax(1, keepdim=True)
        best_score = forward_var.gather(1, backward_indices).squeeze(1)

        best_path = [backward_indices]
        for max_indices in reversed(backpointers):
            # [batch_size]
            backward_indices = max_indices.gather(1, backward_indices)
            best_path.append(backward_indices)
        best_path = torch.cat(best_path, dim=1).flip(1)

        return best_path

    def forward(self, features):
        return self._viterbi_decode(features)


class XLNetTaggingModel(torch.nn.Module):

    def __init__(self, embed, hidden_size, num_labels, class_weight=None, dropout_rate=0.1, ignore_index=-100,
                 use_crf=True):
        torch.nn.Module.__init__(self)
        self.pretrained_model = embed
        self.memories = None
        self.num_labels = num_labels
        self.ignore_index = ignore_index

        self.ff_output = torch.nn.Linear(hidden_size, num_labels, bias=False)
        self.ff_dropout = torch.nn.Dropout(dropout_rate)

        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)

        self.use_crf = use_crf
        self.crf = LinearCRF(num_labels=num_labels) if self.use_crf else None

    def compute_loss(self, input, target):
        if self.use_crf:
            return self.crf.neg_log_likelihood(
                input=input,
                target=target,
                reduction="mean",
                weight=self.class_weight)
        else:
            return torch.nn.functional.cross_entropy(
                input=input.view(-1, input.shape[2]),
                target=target.flatten(),
                weight=self.class_weight,
                ignore_index=self.ignore_index,
                reduction="mean")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            token_type_ids=None,
            input_mask=None,
            use_cache=True,
            label_ids=None
    ):
        # pretrained model outputs
        pretrained_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=mems or self.memories,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            use_cache=use_cache)

        if len(pretrained_output) > 1:
            self.memories = pretrained_output[1]

        x = self.ff_dropout(pretrained_output[0])
        x = self.ff_output(x)

        if label_ids is None:
            # decoding outputs, viterbi for crf and greed argmax for cross entropy.
            # output: [batch_size, seq_len, num_labels]
            if self.use_crf:
                pred_label_ids, pred_scores = self.crf(x)
            else:
                pred_label_ids = torch.argmax(x, dim=2)
            output = (pred_label_ids, x)
        else:
            loss = self.compute_loss(input=x, target=label_ids)
            output = (loss, x)

        return output
