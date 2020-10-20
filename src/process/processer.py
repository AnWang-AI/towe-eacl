import torch
import numpy as np
import os

from functools import partial

from src.tools.utils import tprint, init_w2v_matrix
from src.tools.TOWE_utils import load_text_target_label, split_dev, numericalize, numericalize_label
from src.process.grapher import Grapher

from torch.nn.utils.rnn import pad_sequence

from torch_geometric.data import Data

from transformers import BertTokenizer

import sys

sys.path.append('..')


class Processer():
    def __init__(self, data_path, word_emb_mode="w2v", build_graph=True):
        self.set_random_seed()

        self.data_path = data_path
        train_file_name = "train.tsv"
        test_file_name = "test.tsv"

        self.train_data_path = os.path.join(data_path, train_file_name)
        self.test_data_path = os.path.join(data_path, test_file_name)

        assert word_emb_mode in ["w2v", "bert"]
        self.word_emb_mode = word_emb_mode
        if word_emb_mode == "w2v":
            # w2v_name = "word_embedding.txt"
            # self.w2v_path = os.path.join(data_path, w2v_name)

            self.w2v_path = "./data/full_glove.txt"
        else:
            self.pretrained_bert_path = "models/bert-base-uncased"
            # self.pretrained_bert_path = "bert-base-uncased"

        self.tag2id = {'B': 1, 'I': 2, 'O': 0}

        self.build_graph_mode = build_graph
        if build_graph:
            self.grapher = Grapher()

    def set_random_seed(self):
        seed = 999
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == torch.device('cuda'):
            torch.cuda.manual_seed_all(seed)

    def load_data(self):
        tprint("preparing data...")

        if self.word_emb_mode == "w2v":
            self.w2v_matrix, self.vocab_id_map, self.id_vocab_map = init_w2v_matrix(self.w2v_path)
            self.w2v_matrix = torch.from_numpy(np.float32(self.w2v_matrix))
            # print("w2v_matrix.shape: ", self.w2v_matrix.shape)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_bert_path)

        train_text, train_t, train_ow = load_text_target_label(self.train_data_path)
        test_text, test_t, test_ow = load_text_target_label(self.test_data_path)

        train_text, train_t, train_ow, dev_text, dev_t, dev_ow, _, _ = split_dev(train_text, train_t, train_ow)

        # check data
        # print("test_text: %s\ntest_target: %s\ntest_opinion: %s\n" % (test_text[0], test_t[0], test_ow[0]))
        # print("train text: %s\ntrain target: %s\ntrain opinion: %s\n" % (train_text[0], train_t[0], train_ow[0]))

        train_data_size, dev_data_size, test_data_size = len(train_text), len(dev_text), len(test_text)

        train_data = [train_text, train_t, train_ow]
        val_data = [dev_text, dev_t, dev_ow]
        test_data = [test_text, test_t, test_ow]

        self.data = {"train": train_data, "valid": val_data, "test": test_data}

        return self.data

    def process_data(self):
        node_data_dict = dict()
        for dataset_type in self.data.keys():
            assert dataset_type in ["train", "valid", "test"]
            text, target, opinion = self.data[dataset_type][0], self.data[dataset_type][1], self.data[dataset_type][2]
            numericalized_text, numericalized_target, numericalized_label = self.numericalize_data(text, target,
                                                                                                   opinion)

            numericalized_aspects = self.get_aspects(text, numericalized_target)

            mask = numericalized_label != 3
            node_data = self.get_node_data(numericalized_text, numericalized_target, numericalized_label, mask,
                                           numericalized_aspects)
            node_data_dict[dataset_type + "_node"] = node_data

            if self.build_graph_mode:
                edge_list, word_tags_list = self.get_edge(text)
                node_data_dict[dataset_type + "_edge"] = edge_list
                node_data_dict[dataset_type + "_node_tag"] = self.padding(word_tags_list, 100, padding_value=0)

        return node_data_dict

    def get_aspects(self, texts, targets):
        aspects = []
        for (text, target) in zip(texts, targets):
            text = text.split()
            aspect = []
            for idx in range(target.shape[0]):
                if target[idx] == 1 or target[idx] == 2:
                    aspect.append(text[idx])
            aspects.append(aspect)

        if self.word_emb_mode == "w2v":
            numericalized_aspects = [torch.tensor(numericalize(aspect, self.vocab_id_map), dtype=torch.long) for aspect
                                     in aspects]
        else:
            numericalized_aspects = [
                torch.tensor(self.numericalize_text_with_bert(aspect, self.tokenizer), dtype=torch.long) for aspect in
                aspects]

        numericalized_aspects = self.padding(numericalized_aspects, 30, padding_value=0)

        return numericalized_aspects

    def get_edge(self, text_list):
        get_graph = partial(self.grapher.get_graph, graph_type="distance+dep")
        graph_list = list(map(get_graph, text_list))
        edge_list = [graph[0] for graph in graph_list]
        word_tags_list = [graph[1] for graph in graph_list]

        return edge_list, word_tags_list

    def numericalize_data(self, texts, targets, opinions, padding=True):

        if self.word_emb_mode == "w2v":
            numericalized_text = [torch.tensor(numericalize(text, self.vocab_id_map), dtype=torch.long) for text in
                                  texts]
        else:
            numericalized_text = [torch.tensor(self.numericalize_text_with_bert(text, self.tokenizer), dtype=torch.long)
                                  for text in texts]

        numericalized_target = [torch.tensor(numericalize_label(target, self.tag2id), dtype=torch.long) for target in
                                targets]
        numericalized_label = [torch.tensor(numericalize_label(label, self.tag2id), dtype=torch.long) for label in
                               opinions]

        if padding:
            numericalized_text = self.padding(numericalized_text, 100)
            numericalized_target = self.padding(numericalized_target, 100)
            numericalized_label = self.padding(numericalized_label, 100)

        return numericalized_text, numericalized_target, numericalized_label

    def numericalize_text_with_bert(self, text, tokenizer):
        text = text.lower()
        tokens = text.split()
        text_idx = tokenizer.encode(tokens)[1:-1]
        return text_idx

    def padding(self, seq_list, max_length=None, padding_value=3):
        if max_length is None:
            padding_seq_list = pad_sequence(seq_list, padding_value=padding_value).t()
        else:
            # padding to specified length
            seq_list.append(torch.zeros(max_length))
            padding_seq_list = pad_sequence([seq for seq in seq_list], padding_value=padding_value).t()[:-1, :]
        return padding_seq_list

    def get_node_data(self, numericalized_text, numericalized_target, numericalized_opinion, mask, aspect):
        data = Data(text_idx=numericalized_text, target=numericalized_target, opinion=numericalized_opinion, mask=mask,
                    aspect=aspect)
        return data


if __name__ == "__main__":
    path = './data/14res'
    processer = Processer(path, word_emb_mode="w2v", build_graph=True)
    processer.load_data()
    data = processer.process_data()
    print(data.keys())
