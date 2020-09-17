import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv, GraphConv, GATConv, GatedGraphConv, RGCNConv
from torch_geometric.data import Data

class Tag_BiLSTM(torch.nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size):
        super(Tag_BiLSTM, self).__init__()

        self.word_embed_dim = word_embed_dim
        self.output_size = output_size

        self.hidden_size = 256

        self.word_embed = nn.Embedding(vocab_size, word_embed_dim)
        self.tag_embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=4)

        self.BiLSTM = torch.nn.LSTM(self.word_embed_dim + 4, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.fc_sentence = torch.nn.Linear(self.hidden_size * 2, self.output_size)

        self.init_weight()

    def init_weight(self):

        self.tag_embedding.weight = torch.nn.Parameter(torch.eye(4), requires_grad=True)

        for weights in [self.BiLSTM.weight_hh_l0, self.BiLSTM.weight_ih_l0]:
            torch.nn.init.orthogonal_(weights)

        # linear
        torch.nn.init.xavier_normal_(self.fc_sentence.weight)


    def forward(self, batch):
        target_embedding = self.tag_embedding(batch.target)

        x = batch.text_idx
        x = self.word_embed(x)

        x = x.reshape(-1, 100, self.word_embed_dim)
        target_embedding = target_embedding.reshape(-1, 100, 4)

        x = torch.cat([x, target_embedding], dim=-1)
        # sentence: [batch size, time step, embed dim]
        encoded, _ = self.BiLSTM(x)

        output = self.fc_sentence(encoded)

        return output
