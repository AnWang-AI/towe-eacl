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
from src.process.Dataset import TOWEDataset
from src.model.Net import Tag_BiLSTM

from src.tools.utils import MultiFocalLoss, tprint
from src.tools.TOWE_utils import score_BIO

sys.path.append('./')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/intsig/PycharmProject/TOWE-EACL/data/14res')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--load_model_name', type=str, default='')
parser.add_argument('--save_model_name', type=str, default='models/EdgeNet_model.ckpt')
parser.add_argument('--train_log', type=str, default='log/train_log')
parser.add_argument('--val_log', type=str, default='log/val_log')
parser.add_argument('--eval_frequency', type=int, default=5)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()


def load_data(data_path, train_batch_size=1, val_batch_size=1):

    train_dataset = TOWEDataset(data_path, split='train')
    val_dataset = TOWEDataset(data_path, split='valid')
    test_dataset = TOWEDataset(data_path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    data_loader = {'train': train_loader, 'valid': val_loader, 'test': test_loader}

    return data_loader


class Trainer():

    def __init__(self, loader, model, criterion, optimizer, args):
        self.train_loader, self.val_loader, self.test_loader = loader['train'], loader['valid'], loader['test']
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.cuda = True if torch.cuda.is_available() and args.cuda else False
        self.args = args

    def eval(self, detail=False, dataset="valid"):
        # Transfer model mode from train to eval.
        self.model.eval()

        # User GPU.
        if self.cuda:
            self.model.cuda()
            print('Eval by GPU ...')
        else:
            print('Eval by CPU ...')

        assert dataset in ["valid", "test"]
        if dataset == "valid":
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        total_loss = 0.0
        total_num = len(data_loader)

        ys, preds = [], []
        # Eval.
        for datas in tqdm(data_loader):

            all_input_ids = datas.text_idx
            all_target = datas.target
            all_opinion = datas.opinion
            all_mask = datas.mask

            labels = all_opinion

            if self.cuda:
                all_input_ids, all_target, all_opinion, all_mask = \
                    all_input_ids.cuda(), all_target.cuda(), all_opinion.cuda(), all_mask.cuda()
                datas = datas.to(device)

            # model eval.
            with torch.no_grad():
                scores = self.model(datas)

                scores = scores.cpu()
                # scores = torch.masked_select(scores, mask.reshape(-1, 100, 1).expand(-1, 100, 6) > 0)
                scores = scores.to(device)

                scores = scores.view(-1, 4)
                # Calculate loss.

                labels = labels.cpu()
                # labels = torch.masked_select(labels, all_input_mask > 0)
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
        info = 'loss: {:.3f}, IE precision: {:.3f}, IE recall: {:.3f}, IE f1: {:.3f}'.format(loss,
                                                                                           ie_precision, ie_recall, ie_score)
        tprint(info)
        self.save_log(info, self.args.val_log)

        pred_list = [pred.tolist() for pred in preds]
        label_list = [y.tolist() for y in ys]
        score_dict = score_BIO(pred_list, label_list, ignore_index=0)
        BIO_info = 'BIO precision: {:.3f}, BIO recall: {:.3f}, BIO f1: {:.3f}'.format(score_dict["precision"],
                                                                                      score_dict["recall"],
                                                                                      score_dict["f1"])
        tprint(BIO_info)


        print('-' * 40)
        self.model.train()
        return ie_score

    def train(self, best_accuracy=None):

        self.model.train()
        if self.cuda:
            self.model.cuda()

        if best_accuracy is None:
            best_accuracy = 0.0

        total_num = len(self.train_loader)

        pred_list = []
        label_list = []

        for i in range(self.args.epochs):
            epoch_index = i + 1
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

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass.

                scores = self.model(datas)

                scores = scores.cpu()
                scores = torch.masked_select(scores, all_mask.reshape(-1, 100, 1).expand(-1, 100, 4)>0)
                scores = scores.to(device)

                scores = scores.view(-1, 4)
                # Calculate loss.

                labels = labels.cpu()
                labels = torch.masked_select(labels, all_mask>0)
                labels = labels.to(device)

                labels = labels.view(-1)
                batch_loss = self.criterion(scores, labels)
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

            accuracy = self.metric_f1_score(y, pred, detail=False)

            ie_score, ie_precision, ie_recall = self.IE_score(y, pred)

            # Print train info.
            info = 'loss: {:.3f}, IE precision: {:.3f}, IE recall: {:.3f}, IE f1: {:.3f}'.format(loss,
                                                                                                 ie_precision,
                                                                                                 ie_recall, ie_score)
            tprint(info)
            # Save train info to log file.
            self.save_log(info, self.args.train_log)

            pred_list = [pred.tolist() for pred in preds]
            label_list = [y.tolist() for y in ys]
            score_dict = score_BIO(pred_list, label_list, ignore_index=0)
            BIO_info = 'BIO precision: {:.3f}, BIO recall: {:.3f}, BIO f1: {:.3f}'.format(score_dict["precision"],
                                                                                          score_dict["recall"],
                                                                                          score_dict["f1"])
            tprint(BIO_info)

            # Eval every {eval_frequency} train epoch
            if epoch_index % self.args.eval_frequency == 0:
                eval_score = self.eval(detail=True, dataset="valid")
                # Save best model
                if eval_score > best_accuracy:
                    tprint('Best model so far, best accuracy {:.3f} -> {:.3f}'.format(best_accuracy, accuracy))
                    best_accuracy = eval_score
                    self.save_model(epoch_index, loss, best_accuracy, self.args.save_model_name)

        self.load_model(model_path=self.args.save_model_name)
        self.eval(detail=True, dataset="test")

    def metric_f1_score(self, y, pred, detail=False):
        f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        if detail:
            tprint('Classification Report: ')
            print(classification_report(y, pred))
            tprint('Total Metric - F1: {:.4f}'.format(f1))
        return f1

    def IE_score(self, y, pred):
        # 计算label不为0的时候的准确率
        total_of_label = 0

        total_of_predict = 0

        TP = 0
        for predict, target in zip(pred, y):
            if predict != 0:
                total_of_predict += 1
            if target != 0:
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
            tprint('Load successful! model saved at {} epoch, best accuracy: {:.3f}, loss: {:.3f}'.format(epoch, best_accuracy,
                                                                                                         loss))
        else:
            tprint('Train from beginning ...')
        return best_accuracy

    def save_log(self, info, log_file):
        with open(log_file, 'a+') as f:
            f.writelines('{}\n'.format(info))

if __name__ == "__main__":

    loader = load_data(args.data_path, args.train_batch_size, args.val_batch_size)
    model = Tag_BiLSTM(word_embed_dim=300, output_size=4, vocab_size=7000)

    # print(model)

    loss_op = torch.nn.CrossEntropyLoss()
    # loss_op = torch.nn.NLLLoss()
    # loss_op = MultiFocalLoss(num_class=6, gamma=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(loader, model, loss_op, optimizer, args)
    trainer.load_model()
    trainer.train()