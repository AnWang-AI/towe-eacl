import os
import json
import argparse
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

from src.process.processer import Processer
from src.process.Dataset import TOWEDataset, TOWEDataset_with_bert, TOWEDataset_with_graph, TOWEDataset_with_graph_with_bert
from src.model.Net import ExtractionNet, ExtractionNet_v2

from src.tools.utils import MultiFocalLoss, tprint
from src.tools.TOWE_utils import score_BIO

from src.model.ConfigParser import Config

sys.path.append('./')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./src/model/conf_bert_gnn_lstm.ini')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--train_batch_size', type=int, default=None)
parser.add_argument('--load_model_name', type=str, default='')
parser.add_argument('--save_model_name', type=str, default='')
parser.add_argument('--eval_frequency', type=int, default=5)
args = parser.parse_args()

config = Config(args.config_path)
config.reset_config(args)
config_dict = config.config_dicts
default_config = config.config_dicts['default']
preprocess_config = config.config_dicts['preprocess']
model_config = config.config_dicts['model']


def set_random_seed():
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(data_path, train_batch_size=1, val_batch_size=1, use_bert=False, build_graph=False):

    if use_bert:
        if build_graph:
            print("use bert! buile graph!")
            train_dataset = TOWEDataset_with_graph_with_bert(data_path, 'train')
            val_dataset = TOWEDataset_with_graph_with_bert(data_path, 'valid')
            test_dataset = TOWEDataset_with_graph_with_bert(data_path, 'test')
        else:
            print("use bert!")
            train_dataset = TOWEDataset_with_bert(data_path, 'train')
            val_dataset = TOWEDataset_with_bert(data_path, 'valid')
            test_dataset = TOWEDataset_with_bert(data_path, 'test')
    else:
        if build_graph:
            print("use w2v! buile graph!")
            train_dataset = TOWEDataset_with_graph(data_path, 'train')
            val_dataset = TOWEDataset_with_graph(data_path, 'valid')
            test_dataset = TOWEDataset_with_graph(data_path, 'test')
        else:
            print("use w2v")
            train_dataset = TOWEDataset(data_path, 'train')
            val_dataset = TOWEDataset(data_path, 'valid')
            test_dataset = TOWEDataset(data_path, 'test')

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    data_loader = {'train': train_loader, 'valid': val_loader, 'test': test_loader}

    return data_loader


class Trainer():

    def __init__(self, loader, model, criterion, optimizer, args, config):
        self.args = args
        self.config = config
        self.default_config = config.config_dicts['default']
        self.preprocess_config = config.config_dicts['preprocess']
        self.model_config = config.config_dicts['model']
        self.cuda = True if torch.cuda.is_available() and self.model_config['cuda'] else False
        self.train_loader, self.val_loader, self.test_loader = loader['train'], loader['valid'], loader['test']
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

    def eval(self, detail=False, dataset="valid"):
        # Transfer model mode from train to eval.
        self.model.eval()

        # User GPU.
        # if self.cuda:
        #     self.model.cuda()
        #     print('Eval by GPU ...')
        # else:
        #     print('Eval by CPU ...')

        assert dataset in ["valid", "test"]
        if dataset == "valid":
            print("'Eval by Valid ...'")
            data_loader = self.val_loader
        else:
            print("'Eval by Test ...'")
            data_loader = self.test_loader

        total_loss = 0.0
        total_num = len(data_loader)

        ys, preds = [], []
        # Eval.
        for datas in data_loader:

            all_input_ids = datas.text_idx
            all_target = datas.target
            all_opinion = datas.opinion
            all_mask = datas.mask

            if self.cuda:
                all_input_ids, all_target, all_opinion = \
                    all_input_ids.cuda(), all_target.cuda(), all_opinion.cuda()
                datas = datas.to(device)

            labels = all_opinion

            # model eval.
            with torch.no_grad():
                scores = self.model(datas)

                scores = scores.cpu()
                scores = torch.masked_select(scores.reshape(-1, 100, num_class), all_mask.reshape(-1, 100, 1).expand(-1, 100, num_class) > 0)
                scores = scores.to(device)

                scores = scores.view(-1, num_class)
                # Calculate loss.

                labels = labels.cpu()
                labels = torch.masked_select(labels, all_mask > 0)
                labels = labels.to(device)

                labels = labels.view(-1)

                # Calculate loss.
                batch_loss = self.criterion(scores, labels)
                # conbine result of epoch to eval
                ys.append(labels)
                preds.append(torch.argmax(scores, dim=1))
                # Count loss and correct.
                total_loss += batch_loss

        y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()

        loss = total_loss / total_num
        accuracy = self.metric_f1_score(y, pred, detail)
        ie_score, ie_precision, ie_recall = self.IE_score(y, pred)

        # Print train info.
        info = 'loss: {:.4f}, IE precision: {:.4f}, IE recall: {:.4f}, IE f1: {:.4f}'.format(loss,
                                                                                           ie_precision, ie_recall, ie_score)
        tprint(info)

        pred_list = [pred.tolist() for pred in preds]
        label_list = [y.tolist() for y in ys]
        score_dict = score_BIO(pred_list, label_list, ignore_index=3)
        BIO_score = score_dict["f1"]
        BIO_info = 'BIO precision: {:.4f}, BIO recall: {:.4f}, BIO f1: {:.4f}'.format(score_dict["precision"],
                                                                                      score_dict["recall"],
                                                                                      score_dict["f1"])
        tprint(BIO_info)
        # Save train info to log file.
        self.save_log(BIO_info, self.model_config['val_log'])


        print('-' * 40)
        self.model.train()
        return BIO_score

    def train(self, best_accuracy=None):

        self.model.train()
        if self.cuda:
            self.model.cuda()

        if best_accuracy is None:
            best_accuracy = 0.0

        total_num = len(self.train_loader)

        for i in range(self.model_config['epochs']):
            epoch_index = i + 1

            # 调整学习率以便开启bert的训练
            trian_bert = False
            if self.default_config['use_bert']:  ## 使用bert的时候
                start_to_train_bert_epoch = 10
                if i >= start_to_train_bert_epoch:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 1e-5
                    trian_bert = True
                else:
                    trian_bert = False
            # else:   ## 不适用bert的时候
            #     start_to_train_word_emb = 40
            #     if i > start_to_train_word_emb:
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] = 1e-5
            #         self.model.word_embed.weight.requires_grad = True


            total_loss = 0.0
            total_corect = 0.0
            ys, preds = [], []
            for idx, datas in enumerate(tqdm(self.train_loader)):

                all_input_ids = datas.text_idx
                all_target = datas.target
                all_opinion = datas.opinion
                all_mask = datas.mask

                if self.cuda:
                    all_input_ids, all_target, all_opinion = \
                        all_input_ids.cuda(), all_target.cuda(), all_opinion.cuda()
                    datas = datas.to(device)

                labels = all_opinion

                # Forward pass.

                scores = self.model(datas, trian_bert)


                scores = scores.cpu()
                scores = torch.masked_select(scores.reshape(-1, 100, num_class), all_mask.reshape(-1, 100, 1).expand(-1, 100, num_class)>0)
                scores = scores.to(device)

                scores = scores.view(-1, num_class)
                # Calculate loss.

                labels = labels.cpu()
                labels = torch.masked_select(labels, all_mask>0)
                labels = labels.to(device)

                labels = labels.view(-1)

                # print(scores.shape)
                # print(labels.shape)

                batch_loss = self.criterion(scores, labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Backward pass.
                batch_loss.backward()
                # Update parameters.
                self.optimizer.step()

                # conbine result of epoch to eval
                ys.append(labels)
                preds.append(torch.argmax(scores, dim=1))

                # Count loss and accuracy
                total_loss += batch_loss

            y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()

            # Epoch average loss and accuracy.
            loss = total_loss / total_num

            # accuracy = self.metric_f1_score(y, pred, detail=False)

            ie_score, ie_precision, ie_recall = self.IE_score(y, pred)

            # Print train info.
            info = 'Train: Epoch: {}, loss: {:.4f}, IE precision: {:.4f}, IE recall: {:.4f}, IE f1: {:.4f}'.format(epoch_index, loss,
                                                                                                 ie_precision,
                                                                                                 ie_recall, ie_score)
            tprint(info)

            pred_list = [pred.tolist() for pred in preds]
            label_list = [y.tolist() for y in ys]
            score_dict = score_BIO(pred_list, label_list, ignore_index=3)
            BIO_info = 'Train: BIO precision: {:.4f}, BIO recall: {:.4f}, BIO f1: {:.4f}'.format(score_dict["precision"],
                                                                                          score_dict["recall"],
                                                                                          score_dict["f1"])
            tprint(BIO_info)

            # Save train info to log file.
            self.save_log(BIO_info, self.model_config['train_log'])

            # Eval every {eval_frequency} train epoch
            if epoch_index % self.args.eval_frequency == 0:
                eval_score = self.eval(detail=False, dataset="valid")
                self.eval(detail=False, dataset="test")
                # Save best model
                if eval_score > best_accuracy:
                    tprint('Best model so far, best eval_score {:.4f} -> {:.4f}'.format(best_accuracy, eval_score))
                    best_accuracy = eval_score
                    self.save_model(epoch_index, loss, eval_score, self.args.save_model_name)

        self.config.print_config()
        self.load_model(model_path=self.model_config['save_model_name'])
        self.eval(detail=True, dataset="test")

    def metric_f1_score(self, y, pred, detail=False):
        f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        if detail:
            tprint('Classification Report: ')
            print(classification_report(y, pred))
            tprint('Total Metric - F1: {:.4f}'.format(f1))
        return f1

    def IE_score(self, y, pred):
        # 计算label不为0和1的时候的准确率
        total_of_label = 0

        total_of_predict = 0

        TP = 0
        for predict, target in zip(pred, y):
            if predict not in [0, 3]:
                total_of_predict += 1
            if target not in [0, 3]:
                total_of_label += 1
                if predict == target:
                    TP += 1

        if total_of_label != 0:
            precision = TP / total_of_label
        else:
            precision = 0
        if total_of_predict != 0:
            recall = TP / total_of_predict
        else:
            recall = 0

        if precision > 0 or recall > 0:
            ie_score = 2*precision*recall/(precision+recall)
        else:
            ie_score = 0

        return ie_score, precision, recall

    def save_model(self, epoch, loss, best_accuracy, save_name):
        # model from GPU mode to CPU mode
        if self.cuda:
            self.model.cpu()
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'best_accuracy': best_accuracy,
        }
        torch.save(ckpt, save_name)
        # model from CPU to GPU for step training
        if self.cuda:
            self.model.cuda()
        return best_accuracy

    def load_model(self, model_path=None):

        if model_path:
            model_path = model_path
        else:
            model_path = self.args.load_model_name

        best_accuracy = None
        if os.path.exists(model_path):
            tprint('Model load state dict from {}'.format(model_path))
            ckpt = torch.load(model_path)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
            # self.model.ienet.load_state_dict(ckpt['model_state_dict'])
            epoch = ckpt['epoch']
            loss = ckpt['loss']
            best_accuracy = ckpt['best_accuracy']
            tprint('Load successful! model saved at {} epoch, best accuracy: {:.4f}, loss: {:.4f}'.format(epoch, best_accuracy,
                                                                                                         loss))
        else:
            tprint('Train from beginning ...')
        return best_accuracy

    def save_log(self, info, log_file):
        with open(log_file, 'a+') as f:
            f.writelines('{}\n'.format(info))

if __name__ == "__main__":

    num_class = 4

    set_random_seed()

    loader = load_data(preprocess_config['data_path'],
                       model_config['train_batch_size'],
                       model_config['val_batch_size'],
                       default_config['use_bert'],
                       default_config['build_graph'])

    if default_config['use_bert']:
        word_embed_dim = 768
        word_emb_mode = "bert"
    else:
        word_embed_dim = 300
        word_emb_mode = "w2v"

    model_name = model_config['model']
    model = eval(model_name)(word_embed_dim=word_embed_dim,
                             output_size=num_class,
                             config_dicts=config_dict,
                             word_emb_mode=word_emb_mode,
                             graph_mode=default_config['build_graph'])

    print(model)

    config.print_config()

    assert model_config['loss'] in ["CrossEntropy", "FacalLoss"]
    if model_config['loss'] == "CrossEntropy":
        # loss_op = torch.nn.CrossEntropyLoss()
        loss_op = torch.nn.NLLLoss()
    else:
        loss_op = MultiFocalLoss(num_class=num_class, gamma=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    trainer = Trainer(loader, model, loss_op, optimizer, args, config)
    trainer.load_model()
    trainer.train()