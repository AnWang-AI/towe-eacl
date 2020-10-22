import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv, GraphConv, GATConv, GatedGraphConv, RGCNConv
from torch_geometric.data import Data

from transformers import BertModel

from src.model.layers.ARGCN_dep_conv import ARGCN_dep_conv
from src.model.layers.ARGCN_dep_distance_conv import ARGCN_dep_distance_conv, ARGCN_dep_distance_conv_v2, \
    ARGCN_dep_distance_conv_multi_head, ARGCN_dep_distance_conv_multi_head_v2
from src.model.layers.ARGCN_distance_conv import ARGCN_distance_conv_multi_head
from src.model.layers.RGAT_conv import RGAT_conv
from src.model.layers.SelfAttention import SelfAttention
from src.model.LSTM_CRF import LinearCRF

from src.tools.utils import init_w2v_matrix


class ExtractionNet(torch.nn.Module):
    def __init__(self, word_embed_dim, output_size, config_dicts, word_emb_mode="w2v", graph_mode=False):
        super(ExtractionNet, self).__init__()

        self.default_config = config_dicts['default']
        self.preprocess_config = config_dicts['preprocess']
        self.model_config = config_dicts['model']

        self.word_embed_dim = word_embed_dim
        self.output_size = output_size

        self.have_word_emb = self.model_config['have_word_emb']

        self.feature_dim = 0

        if self.have_word_emb:
            self.word_emb_mode = word_emb_mode
            assert word_emb_mode in ["w2v", "bert"]
            if word_emb_mode == "w2v":

                # w2v_path = "./data/14res/word_embedding.txt"
                w2v_path = self.preprocess_config['w2v_path']

                self.w2v_matrix, self.vocab_id_map, self.id_vocab_map = init_w2v_matrix(w2v_path)
                self.w2v_matrix = torch.from_numpy(np.float32(self.w2v_matrix))
                vocab_size = self.preprocess_config['vocab_size']
                self.word_embed = nn.Embedding(vocab_size, word_embed_dim)

            else:
                bert_path = self.preprocess_config['pretrained_bert_path']
                self.embedding_model = BertModel.from_pretrained(bert_path)

            self.feature_dim += self.word_embed_dim

        self.target_emb_dim = self.model_config['target_embedding_dim']
        self.target_embedding = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=self.target_emb_dim)
        self.feature_dim += self.target_emb_dim

        self.have_tag = self.model_config['have_tag']
        if self.have_tag:
            self.tag_emb_dim = 100
            self.tag_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.tag_emb_dim)
            self.feature_dim += self.tag_emb_dim

        self.hidden_size = self.model_config['hidden_size']

        self.graph_mode = graph_mode

        if graph_mode == True:
            mainnet_name = self.model_config['mainnet']
            self.MainNet = eval(mainnet_name)(num_features=self.feature_dim, num_classes=self.hidden_size)

            if self.have_word_emb:
                self.LSTM_input_dim = self.hidden_size + self.word_embed_dim
            else:
                self.LSTM_input_dim = self.hidden_size
            self.SubNet = BiLSTMNet(input_dim=self.LSTM_input_dim, ouput_dim=output_size,
                                    hidden_size=self.hidden_size)

            # self.MainNet = DeepARGCNNet(num_features=self.feature_dim, num_classes=output_size)

        else:
            self.MainNet = BiLSTMNet(input_dim=self.feature_dim, ouput_dim=output_size,
                                     hidden_size=self.hidden_size)

        self.init_weight()

    def init_weight(self):
        if self.have_word_emb:
            if self.word_emb_mode == "w2v":
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.word_embed.weight = torch.nn.Parameter(self.w2v_matrix.to(device), requires_grad=False)

        # self.tag_embedding.weight = torch.nn.Parameter(torch.eye(4), requires_grad=True)
        torch.nn.init.xavier_normal_(self.target_embedding.weight)

        if self.have_tag:
            torch.nn.init.xavier_normal_(self.tag_embedding.weight)

    def forward(self, batch, trian_bert=False):

        target_embedding = self.target_embedding(batch.target)
        target_embedding = target_embedding.reshape(-1, 100, self.target_emb_dim)

        x = batch.text_idx

        if self.have_word_emb:
            if self.word_emb_mode == "w2v":
                word_embedding = self.word_embed(x)
                word_embedding = word_embedding.reshape(-1, 100, self.word_embed_dim)
            else:
                x = x.reshape(-1, 100)
                # bert_model_input_size: [batch size, time step]
                if trian_bert:
                    word_embedding = self.embedding_model(x)[0]
                else:
                    with torch.no_grad():
                        word_embedding = self.embedding_model(x)[0]

            x = torch.cat([word_embedding, target_embedding], dim=-1)
        else:
            x = target_embedding

        if self.have_tag:
            tag_embedding = self.tag_embedding(batch.tag)
            tag_embedding = tag_embedding.reshape(-1, 100, self.tag_emb_dim)
            x = torch.cat([x, tag_embedding], dim=-1)

        if self.graph_mode:
            x = x.reshape(-1, self.feature_dim)
            # x = x.reshape(-1, self.word_embed_dim + self.target_emb_dim + self.tag_embed_dim)
            edge_idx = batch.edge_index
            edge_type = batch.edge_type
            edge_distance = batch.edge_distance

            x = self.MainNet(x, edge_idx, edge_type, edge_distance)
            x = x.reshape(-1, 100, self.hidden_size)

            if self.have_word_emb:
                x = torch.cat([x, word_embedding], dim=-1)

            x = self.SubNet(x)
        else:
            # x shape: [batch size, time step, embed dim]
            x = self.MainNet(x)

        output = F.log_softmax(x, dim=-1)

        return output


class ExtractionNet_crf(torch.nn.Module):
    def __init__(self, word_embed_dim, output_size, config_dicts, word_emb_mode="w2v", graph_mode=False):
        super(ExtractionNet_crf, self).__init__()

        self.default_config = config_dicts['default']
        self.preprocess_config = config_dicts['preprocess']
        self.model_config = config_dicts['model']

        self.word_embed_dim = word_embed_dim
        self.output_size = output_size

        self.have_word_emb = self.model_config['have_word_emb']

        self.feature_dim = 0

        if self.have_word_emb:
            self.word_emb_mode = word_emb_mode
            assert word_emb_mode in ["w2v", "bert"]
            if word_emb_mode == "w2v":

                # w2v_path = "./data/14res/word_embedding.txt"
                w2v_path = self.preprocess_config['w2v_path']

                self.w2v_matrix, self.vocab_id_map, self.id_vocab_map = init_w2v_matrix(w2v_path)
                self.w2v_matrix = torch.from_numpy(np.float32(self.w2v_matrix))
                vocab_size = self.preprocess_config['vocab_size']
                self.word_embed = nn.Embedding(vocab_size, word_embed_dim)

            else:
                bert_path = self.preprocess_config['pretrained_bert_path']
                self.embedding_model = BertModel.from_pretrained(bert_path)

            self.feature_dim += self.word_embed_dim

        self.target_emb_dim = self.model_config['target_embedding_dim']
        self.target_embedding = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=self.target_emb_dim)
        self.feature_dim += self.target_emb_dim

        self.have_tag = self.model_config['have_tag']
        if self.have_tag:
            self.tag_emb_dim = 100
            self.tag_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.tag_emb_dim)
            self.feature_dim += self.tag_emb_dim

        self.hidden_size = self.model_config['hidden_size']

        self.graph_mode = graph_mode

        if graph_mode == True:

            self.SubNet1 = BiLSTMNet(input_dim=self.feature_dim, ouput_dim=self.hidden_size,
                                     hidden_size=self.hidden_size)

            mainnet_name = self.model_config['mainnet']
            self.MainNet = eval(mainnet_name)(num_features=self.hidden_size, num_classes=self.hidden_size)

            self.LSTM_input_dim = self.hidden_size + self.word_embed_dim

            self.SubNet2 = BiLSTMNet(input_dim=self.LSTM_input_dim, ouput_dim=output_size,
                                     hidden_size=self.hidden_size)

            # self.MainNet = DeepARGCNNet(num_features=self.feature_dim, num_classes=output_size)

        else:
            self.MainNet = BiLSTMNet(input_dim=self.feature_dim, ouput_dim=output_size,
                                     hidden_size=self.hidden_size)

        self.crf = LinearCRF(num_labels=output_size)

        self.init_weight()

    def init_weight(self):
        if self.have_word_emb:
            if self.word_emb_mode == "w2v":
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.word_embed.weight = torch.nn.Parameter(self.w2v_matrix.to(device), requires_grad=False)

        # self.tag_embedding.weight = torch.nn.Parameter(torch.eye(4), requires_grad=True)
        torch.nn.init.xavier_normal_(self.target_embedding.weight)

        if self.have_tag:
            torch.nn.init.xavier_normal_(self.tag_embedding.weight)

    def forward(self, batch, trian_bert=False):

        target_embedding = self.target_embedding(batch.target)
        target_embedding = target_embedding.reshape(-1, 100, self.target_emb_dim)

        x = batch.text_idx

        if self.have_word_emb:
            if self.word_emb_mode == "w2v":
                word_embedding = self.word_embed(x)
                word_embedding = word_embedding.reshape(-1, 100, self.word_embed_dim)
            else:
                x = x.reshape(-1, 100)
                # bert_model_input_size: [batch size, time step]
                if trian_bert:
                    word_embedding = self.embedding_model(x)[0]
                else:
                    with torch.no_grad():
                        word_embedding = self.embedding_model(x)[0]

            x = torch.cat([word_embedding, target_embedding], dim=-1)
        else:
            x = target_embedding

        if self.have_tag:
            tag_embedding = self.tag_embedding(batch.tag)
            tag_embedding = tag_embedding.reshape(-1, 100, self.tag_emb_dim)
            x = torch.cat([x, tag_embedding], dim=-1)

        if self.graph_mode:
            x = self.SubNet1(x)

            x = x.reshape(-1, self.hidden_size)
            # x = x.reshape(-1, self.word_embed_dim + self.target_emb_dim + self.tag_embed_dim)
            edge_idx = batch.edge_index
            edge_type = batch.edge_type
            edge_distance = batch.edge_distance

            x = self.MainNet(x, edge_idx, edge_type, edge_distance)
            x = x.reshape(-1, 100, self.hidden_size)
            x = torch.cat([x, word_embedding], dim=-1)

            x = self.SubNet2(x)
        else:
            # x shape: [batch size, time step, embed dim]
            x = self.MainNet(x)

        output = x
        # output = F.log_softmax(x, dim=-1)

        return output


class ExtractionNet_mrc(torch.nn.Module):
    def __init__(self, word_embed_dim, output_size, config_dicts, word_emb_mode="w2v", graph_mode=False):
        super(ExtractionNet_mrc, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.default_config = config_dicts['default']
        self.preprocess_config = config_dicts['preprocess']
        self.model_config = config_dicts['model']

        self.word_embed_dim = word_embed_dim
        self.output_size = output_size

        self.have_word_emb = self.model_config['have_word_emb']

        self.feature_dim = 0

        if self.have_word_emb:
            self.word_emb_mode = word_emb_mode
            assert word_emb_mode in ["w2v", "bert"]
            if word_emb_mode == "w2v":

                # w2v_path = "./data/14res/word_embedding.txt"
                w2v_path = self.preprocess_config['w2v_path']

                self.w2v_matrix, self.vocab_id_map, self.id_vocab_map = init_w2v_matrix(w2v_path)
                self.w2v_matrix = torch.from_numpy(np.float32(self.w2v_matrix))
                vocab_size = self.preprocess_config['vocab_size']
                self.word_embed = nn.Embedding(vocab_size, word_embed_dim)

            else:
                bert_path = self.preprocess_config['pretrained_bert_path']
                self.embedding_model = BertModel.from_pretrained(bert_path)

            self.feature_dim += self.word_embed_dim

        self.target_emb_dim = self.model_config['target_embedding_dim']
        self.target_embedding = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=self.target_emb_dim)
        self.feature_dim += self.target_emb_dim

        # self.feature_dim += self.word_embed_dim

        self.have_tag = self.model_config['have_tag']
        if self.have_tag:
            self.tag_emb_dim = 100
            self.tag_embedding = torch.nn.Embedding(num_embeddings=50, embedding_dim=self.tag_emb_dim)
            self.feature_dim += self.tag_emb_dim

        self.hidden_size = self.model_config['hidden_size']
        self.q_lin = torch.nn.Linear(self.word_embed_dim, self.hidden_size)

        self.graph_mode = graph_mode

        if graph_mode == True:
            mainnet_name = self.model_config['mainnet']
            self.MainNet = eval(mainnet_name)(num_features=self.feature_dim, num_classes=self.hidden_size)

            if self.have_word_emb:
                self.LSTM_input_dim = self.hidden_size + self.word_embed_dim
            else:
                self.LSTM_input_dim = self.hidden_size
            self.SubNet = BiLSTMNet(input_dim=self.LSTM_input_dim, ouput_dim=self.hidden_size,
                                    hidden_size=self.hidden_size)

        else:
            self.MainNet = BiLSTMNet(input_dim=self.feature_dim, ouput_dim=self.hidden_size,
                                     hidden_size=self.hidden_size)

        self.self_att = SelfAttention(hidden_size=2 * self.hidden_size, num_attention_heads=8, dropout_prob=0.2)

        self.fin_lin = torch.nn.Linear(2 * self.hidden_size, output_size)

        # self.fin_net = BiLSTMNet(input_dim=2 * self.hidden_size, ouput_dim=output_size,
        #                         hidden_size=self.hidden_size, bidirectional=False)

        self.init_weight()

    def init_weight(self):
        if self.have_word_emb:
            if self.word_emb_mode == "w2v":
                self.word_embed.weight = torch.nn.Parameter(self.w2v_matrix.to(self.device), requires_grad=False)

        # self.tag_embedding.weight = torch.nn.Parameter(torch.eye(4), requires_grad=True)
        torch.nn.init.xavier_normal_(self.target_embedding.weight)

        if self.have_tag:
            torch.nn.init.xavier_normal_(self.tag_embedding.weight)

        torch.nn.init.xavier_normal_(self.q_lin.weight)
        torch.nn.init.xavier_normal_(self.fin_lin.weight)

    def forward(self, batch, trian_bert=False):

        target_embedding = self.target_embedding(batch.target)
        target_embedding = target_embedding.reshape(-1, 100, self.target_emb_dim)

        x = batch.text_idx

        if self.have_word_emb:
            if self.word_emb_mode == "w2v":
                word_embedding = self.word_embed(x)
                word_embedding = word_embedding.reshape(-1, 100, self.word_embed_dim)
            else:
                x = x.reshape(-1, 100)
                # bert_model_input_size: [batch size, time step]
                if trian_bert:
                    word_embedding = self.embedding_model(x)[0]
                else:
                    with torch.no_grad():
                        word_embedding = self.embedding_model(x)[0]

            x = torch.cat([word_embedding, target_embedding], dim=-1)
        else:
            x = target_embedding


        aspect = batch.aspect
        if self.word_emb_mode == "w2v":
            aspect_embedding = self.word_embed(aspect)
            aspect_embedding = aspect_embedding.reshape(-1, 30, self.word_embed_dim)
            aspect = aspect.reshape(-1, 30)
        else:
            aspect = aspect.reshape(-1, 30)
            if trian_bert:
                aspect_embedding = self.embedding_model(aspect)[0]
            else:
                with torch.no_grad():
                    aspect_embedding = self.embedding_model(aspect)[0]

        aspect_length = (aspect>0).sum(-1).reshape(-1, 1)

        question_embedding = aspect_embedding.sum(axis=1)/aspect_length
        # question_embedding = aspect_embedding.max(axis=1).values
        # question_embedding = question_embedding/10

        question_embedding = question_embedding.unsqueeze(dim=1)
        question_embedding = question_embedding.expand(question_embedding.shape[0], 100, question_embedding.shape[2])
        question_embedding = F.relu(question_embedding)
        question_rep = self.q_lin(question_embedding)
        question_rep = F.relu(question_rep)

        if self.have_tag:
            tag_embedding = self.tag_embedding(batch.tag)
            tag_embedding = tag_embedding.reshape(-1, 100, self.tag_emb_dim)
            x = torch.cat([x, tag_embedding], dim=-1)

        if self.graph_mode:
            x = x.reshape(-1, self.feature_dim)
            # x = x.reshape(-1, self.word_embed_dim + self.target_emb_dim + self.tag_embed_dim)
            edge_idx = batch.edge_index
            edge_type = batch.edge_type
            edge_distance = batch.edge_distance

            x = self.MainNet(x, edge_idx, edge_type, edge_distance)
            x = x.reshape(-1, 100, self.hidden_size)

            if self.have_word_emb:
                x = torch.cat([x, word_embedding], dim=-1)

            x = self.SubNet(x)
        else:
            # x shape: [batch size, time step, embed dim]
            x = self.MainNet(x)

        x = F.relu(x)

        x = torch.cat([x, question_rep], dim=-1)
        # print(batch.aspect.reshape(-1, 30))
        # print(x.shape)
        # print(torch.ones(x.shape).cuda().shape)
        x = self.self_att(x, torch.ones(x.shape[:2]).to(self.device))+x
        x = F.relu(x)
        x = self.fin_lin(x)

        output = F.log_softmax(x, dim=-1)

        return output


class BiLSTMNet(torch.nn.Module):
    def __init__(self, input_dim, ouput_dim, hidden_size, bidirectional=True):
        super(BiLSTMNet, self).__init__()

        self.BiLSTM = torch.nn.LSTM(input_dim, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.lin = torch.nn.Linear(hidden_size * 2, ouput_dim)
        else:
            self.lin = torch.nn.Linear(hidden_size, ouput_dim)

        # self.BiLSTM = torch.nn.LSTM(num_features, num_classes, num_layers=1, bidirectional=False, batch_first=True)

        self.init_weight()

    def init_weight(self):
        # lstm
        for weights in [self.BiLSTM.weight_hh_l0, self.BiLSTM.weight_ih_l0]:
            torch.nn.init.orthogonal_(weights)

        # linear
        torch.nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x):
        x, _ = self.BiLSTM(x)
        x = self.lin(x)

        return x


class ARGCNNet(torch.nn.Module):

    def __init__(self, num_features=768, num_classes=9, edge_feature_dim=100):
        super(ARGCNNet, self).__init__()

        self.num_features = num_features

        self.hidden_dim = 256

        conv_layer = ARGCN_dep_distance_conv

        self.conv1 = conv_layer(num_features, self.hidden_dim, edge_feature_dim=edge_feature_dim)

        self.conv2 = conv_layer(self.hidden_dim, num_classes, edge_feature_dim=edge_feature_dim)

    def forward(self, x, edge_index, edge_type, edge_distance):
        x = self.conv1(x, edge_index, edge_type, edge_distance)
        x = F.dropout(x, p=0.4)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_type, edge_distance)
        x = F.dropout(x, p=0.4)
        x = F.relu(x)

        return x


class RGCNNet(torch.nn.Module):

    def __init__(self, num_features=768, num_classes=9, edge_feature_dim=100):
        super(RGCNNet, self).__init__()

        self.num_features = num_features

        self.hidden_dim = 256

        conv_layer = RGCNConv

        self.conv1 = conv_layer(num_features, self.hidden_dim, num_relations=50, num_bases=1)

        self.conv2 = conv_layer(self.hidden_dim, num_classes, num_relations=50, num_bases=1)

    def forward(self, x, edge_index, edge_type, edge_distance):
        edge_type = edge_type.reshape(-1)
        x = self.conv1(x, edge_index, edge_type)
        x = F.dropout(x, p=0.4)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_type)
        x = F.dropout(x, p=0.4)
        x = F.relu(x)

        return x


class DeepARGCNNet(torch.nn.Module):
    def __init__(self, num_features=768, num_classes=9, edge_feature_dim=2, num_mid_layers=3):
        super(DeepARGCNNet, self).__init__()

        self.num_features = num_features
        self.num_mid_layers = num_mid_layers

        self.norm_layer_list = torch.nn.ModuleList()
        self.conv_layer_list = torch.nn.ModuleList()

        conv_layer = ARGCN_dep_distance_conv_multi_head
        # conv_layer = RGAT_conv
        # conv_layer = ARGCN_distance_conv_multi_head

        self.hidden_dim = 128

        # self.norm_layer_list.append(torch.nn.LayerNorm(num_features, eps=1e-05))

        self.norm_layer_list.append(torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True))
        self.conv_layer_list.append(conv_layer(num_features, self.hidden_dim, edge_feature_dim=edge_feature_dim))

        for i in range(self.num_mid_layers):
            # self.norm_layer_list.append(torch.nn.LayerNorm(self.hidden_dim, eps=1e-05))
            self.norm_layer_list.append(torch.nn.BatchNorm1d(self.hidden_dim, eps=1e-05, momentum=0.1, affine=True))
            self.conv_layer_list.append(conv_layer(self.hidden_dim, self.hidden_dim, edge_feature_dim=edge_feature_dim))

        self.norm_layer_list.append(torch.nn.BatchNorm1d(self.hidden_dim, eps=1e-05, momentum=0.1, affine=True))
        self.conv_layer_list.append(conv_layer(self.hidden_dim, num_classes, edge_feature_dim=edge_feature_dim))

    def forward(self, x, edge_index, edge_type, edge_distance):

        x = self.conv_layer_list[0](x, edge_index, edge_type, edge_distance)

        for i in range(self.num_mid_layers):
            x = self.norm_layer_list[i + 1](x)
            x = F.leaky_relu(x, 0.1)

            x = self.conv_layer_list[i + 1](x, edge_index, edge_type, edge_distance) + x

        x = self.norm_layer_list[-1](x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv_layer_list[-1](x, edge_index, edge_type, edge_distance)

        x = F.leaky_relu(x, 0.1)

        return x


class DeepRGCNNet(torch.nn.Module):
    def __init__(self, num_features=768, num_classes=9, edge_feature_dim=2, num_mid_layers=3):
        super(DeepRGCNNet, self).__init__()

        self.num_features = num_features
        self.num_mid_layers = num_mid_layers

        self.norm_layer_list = torch.nn.ModuleList()
        self.conv_layer_list = torch.nn.ModuleList()

        conv_layer = RGCNConv

        self.hidden_dim = 128

        # self.norm_layer_list.append(torch.nn.LayerNorm(num_features, eps=1e-05))

        self.norm_layer_list.append(torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True))
        self.conv_layer_list.append(conv_layer(num_features, self.hidden_dim, num_relations=50, num_bases=1))

        for i in range(self.num_mid_layers):
            # self.norm_layer_list.append(torch.nn.LayerNorm(self.hidden_dim, eps=1e-05))
            self.norm_layer_list.append(torch.nn.BatchNorm1d(self.hidden_dim, eps=1e-05, momentum=0.1, affine=True))
            self.conv_layer_list.append(conv_layer(self.hidden_dim, self.hidden_dim, num_relations=50, num_bases=1))

        self.norm_layer_list.append(torch.nn.BatchNorm1d(self.hidden_dim, eps=1e-05, momentum=0.1, affine=True))
        self.conv_layer_list.append(conv_layer(self.hidden_dim, num_classes, num_relations=50, num_bases=1))

    def forward(self, x, edge_index, edge_type, edge_distance):

        x = self.conv_layer_list[0](x, edge_index, edge_type.reshape(-1))

        for i in range(self.num_mid_layers):
            x = self.norm_layer_list[i + 1](x)
            x = F.leaky_relu(x, 0.1)
            x = self.conv_layer_list[i + 1](x, edge_index, edge_type.reshape(-1)) + x

        x = self.norm_layer_list[-1](x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv_layer_list[-1](x, edge_index, edge_type.reshape(-1))

        x = F.leaky_relu(x, 0.1)

        return x


class DeepGATNet(torch.nn.Module):
    def __init__(self, num_features=768, num_classes=9, num_mid_layers=3):
        super(DeepGATNet, self).__init__()

        self.num_features = num_features
        self.num_mid_layers = num_mid_layers

        self.norm_layer_list = torch.nn.ModuleList()
        self.conv_layer_list = torch.nn.ModuleList()

        conv_layer = GATConv

        self.hidden_dim = 128

        # self.norm_layer_list.append(torch.nn.LayerNorm(num_features, eps=1e-05))

        self.norm_layer_list.append(torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True))
        self.conv_layer_list.append(conv_layer(num_features, self.hidden_dim))

        for i in range(self.num_mid_layers):
            # self.norm_layer_list.append(torch.nn.LayerNorm(self.hidden_dim, eps=1e-05))
            self.norm_layer_list.append(torch.nn.BatchNorm1d(self.hidden_dim, eps=1e-05, momentum=0.1, affine=True))
            self.conv_layer_list.append(conv_layer(self.hidden_dim, self.hidden_dim))

        self.norm_layer_list.append(torch.nn.BatchNorm1d(self.hidden_dim, eps=1e-05, momentum=0.1, affine=True))
        self.conv_layer_list.append(conv_layer(self.hidden_dim, num_classes))

    def forward(self, x, edge_index, edge_type, edge_distance):

        x = self.conv_layer_list[0](x, edge_index)

        for i in range(self.num_mid_layers):
            x = self.norm_layer_list[i + 1](x)
            x = F.leaky_relu(x, 0.1)
            x = self.conv_layer_list[i + 1](x, edge_index) + x

        x = self.norm_layer_list[-1](x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv_layer_list[-1](x, edge_index)

        x = F.leaky_relu(x, 0.1)

        # x = F.log_softmax(x, dim=1)

        return x
