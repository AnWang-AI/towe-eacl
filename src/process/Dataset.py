import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from src.process.processer import Processer

class TOWEDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):

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

        processer = Processer(self.root)
        processer.load_data()
        dataset = processer.process_data()

        for s, split in enumerate(['train', 'valid', 'test']):

            current_data = dataset[split]
            data_list = []
            for idx in range(current_data.text_idx.shape[0]):
                text_idx = current_data.text_idx[idx]
                opinion = current_data.opinion[idx]
                target = current_data.target[idx]
                mask = current_data.mask[idx]
                data = Data(text_idx=text_idx, opinion=opinion, target=target, mask=mask)
                data_list.append(data)
            # 这里的save方式以及路径需要对应构造函数中的load操作
            torch.save(self.collate(data_list), self.processed_paths[s])

if __name__ == "__main__":
    path = '/home/intsig/PycharmProject/TOWE-EACL/data/14res'
    train_dataset = TOWEDataset(path, split='train')
