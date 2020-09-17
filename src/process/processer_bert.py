import torch
import numpy as np
import os

from src.tools.utils import tprint, init_w2v_matrix
from src.tools.TOWE_utils import load_text_target_label, split_dev, numericalize, numericalize_label

from torch.nn.utils.rnn import pad_sequence

from torch_geometric.data import Data

import sys
sys.path.append('..')

class Processer():
    def __init__(self, data_path):
        self.data_path = data_path
        train_file_name = "train.tsv"
        test_file_name = "test.tsv"

        self.train_data_path = os.path.join(data_path, train_file_name)
        self.test_data_path = os.path.join(data_path, test_file_name)

        self.tag2id = {'B': 2, 'I': 3, 'O': 1}

    def load_data(self):
        tprint("preparing data...")

        train_text, train_t, train_ow = load_text_target_label(self.train_data_path)
        test_text, test_t, test_ow = load_text_target_label(self.test_data_path)

        # check data
        # print("test_text: %s\ntest_target: %s\ntest_opinion: %s\n" % (test_text[0], test_t[0], test_ow[0]))

        train_text, train_t, train_ow, dev_text, dev_t, dev_ow, _, _ = split_dev(train_text, train_t, train_ow)

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
            numericalized_text, numericalized_target, numericalized_label = self.numericalize_data(text, target, opinion)
            mask = numericalized_label > 0
            node_data = self.get_node_data(numericalized_text, numericalized_target, numericalized_label, mask)
            node_data_dict[dataset_type] = node_data
        return node_data_dict

    def numericalize_data(self, texts, targets, opinions, padding=True):

        numericalized_text = [torch.tensor(numericalize(text, self.vocab_id_map), dtype=torch.long) for text in texts]
        numericalized_target = [torch.tensor(numericalize_label(target, self.tag2id), dtype=torch.long) for target in targets]
        numericalized_label = [torch.tensor(numericalize_label(label, self.tag2id), dtype=torch.long) for label in opinions]

        if padding:
            numericalized_text = self.padding(numericalized_text, 100)
            numericalized_target = self.padding(numericalized_target, 100)
            numericalized_label = self.padding(numericalized_label, 100)

        return numericalized_text, numericalized_target, numericalized_label

    def padding(self, seq_list, max_length=None):
        if max_length is None:
            padding_seq_list = pad_sequence(seq_list).t()
        else:
            # padding to specified length
            seq_list.append(torch.zeros(max_length))
            padding_seq_list = pad_sequence([seq for seq in seq_list]).t()[:-1, :]
        return padding_seq_list


    def get_node_data(self, numericalized_text, numericalized_target, numericalized_opinion, mask):
        data = Data(text_idx=numericalized_text, target=numericalized_target, opinion=numericalized_opinion, mask=mask)
        return data

if __name__ == "__main__":
    path = '/home/intsig/PycharmProject/TOWE-EACL/data/14res'
    processer = Processer(path)
    processer.load_data()
    data = processer.process_data()
    print(data)