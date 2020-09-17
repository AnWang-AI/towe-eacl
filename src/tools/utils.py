import datetime
import functools

import codecs
import numpy as np
from tqdm import tqdm
import pickle

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

def timestamp(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        print("%s | " % datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"), end = "")
        return func(*args, **kwargs)
    return decorated

@timestamp
def tprint(*args, **kwargs):
    print(*args, **kwargs)


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def init_w2v_matrix(w2v_matrix_path, mode=None):
    """
      w2v matrix第n行代表第n个词，但在词表中，id为n+2.
    """
    if mode == None:
        mode = "pickle" if "pkl" in w2v_matrix_path else "text"

    if mode == "text":
        w2v_matrix = []
        vocab_id_map, id_vocab_map = dict(), dict()
        with codecs.open(w2v_matrix_path, encoding="utf8", errors="ignore") as f_in:
            word_number = 0
            vector_dimension = 0
            tprint("loading word embedding matrix...")
            for index, line in tqdm(enumerate(f_in), desc="load word embedding"):
                line_array = [section.strip() for section in line.strip().split(" ")]
                if index == 0:
                    word_number = eval(line_array[0])
                    vector_dimension = eval(line_array[1])
                else:
                    word = line_array[0]
                    vector = [eval(s) for s in line_array[1:]]
                    assert (len(vector) == vector_dimension)
                    vocab_id_map[word] = index + 1  # 从2开始记录w2v, 0/1表示pad/unk
                    id_vocab_map[index + 1] = word
                    w2v_matrix.append(vector)
            assert (len(vocab_id_map) == word_number)
            assert (len(id_vocab_map) == word_number)
            assert (len(w2v_matrix) == word_number)
            vocab_size = word_number + 2
            pad_embedding = np.zeros(shape=[1, vector_dimension])
            unk_embedding = np.random.standard_normal(size=[1, vector_dimension])
        w2v_matrix = np.concatenate([pad_embedding, unk_embedding, np.array(w2v_matrix)], axis=0)
        vocab_id_map["<PAD>"] = 0
        vocab_id_map["<UNK>"] = 1
        id_vocab_map[0] = "<PAD>"
        id_vocab_map[1] = "<UNK>"
    elif mode == "pickle":
        tprint("loading pickle file [%s]..." % '/'.join(w2v_matrix_path.split('/')[-2:]))
        with open(w2v_matrix_path, "rb") as f_read:
            w2v_matrix = pickle.load(f_read)
            vocab_id_map, id_vocab_map = None, None
        tprint("done.")
    else:
        w2v_matrix, vocab_id_map, id_vocab_map = None, None, None
    return w2v_matrix, vocab_id_map, id_vocab_map

