import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from src.process.processer import Processer


class TOWEDataset(InMemoryDataset):
    def __init__(self, root, config, split='train', word_emb_mode='w2v', build_graph=False, transform=None, pre_transform=None, pre_filter=None):

        self.word_emb_mode = word_emb_mode
        self.build_graph = build_graph
        self.config = config

        super(TOWEDataset, self).__init__(root, transform, pre_transform, pre_filter)

        assert split in ['train', 'valid', 'test']

        # 加载数据
        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):


        processer = Processer(self.root, word_emb_mode=self.word_emb_mode, build_graph=self.build_graph, distance_gate=self.config["threshold"])

        processer.load_data()
        dataset = processer.process_data()

        for s, split in enumerate(['train', 'valid', 'test']):

            current_data = dataset[split+"_node"]

            if self.build_graph:
                current_edge_data = dataset[split+"_edge"]
                current_node_tag_data = dataset[split + "_node_tag"]

            data_list = []
            for idx in range(current_data.text_idx.shape[0]):
                text_idx = current_data.text_idx[idx]
                opinion = current_data.opinion[idx]
                target = current_data.target[idx]
                mask = current_data.mask[idx]
                aspect = current_data.aspect[idx]

                if self.build_graph:
                    edge_idx = current_edge_data[idx].edge_idx
                    edge_type = current_edge_data[idx].edge_type
                    tag = current_node_tag_data[idx]

                    # try:
                    edge_distance = current_edge_data[idx].edge_distance
                    data = Data(text_idx=text_idx, opinion=opinion, target=target, mask=mask, aspect=aspect,
                                edge_index=edge_idx, edge_type=edge_type, edge_distance=edge_distance,
                                tag=tag, num_nodes=100)
                else:
                    data = Data(text_idx=text_idx, opinion=opinion, target=target, mask=mask, aspect=aspect)

                data_list.append(data)
            # 这里的save方式以及路径需要对应构造函数中的load操作
            torch.save(self.collate(data_list), self.processed_paths[s])



if __name__ == "__main__":
    path = './data/14res'
    train_dataset = TOWEDataset(path, split='train')
