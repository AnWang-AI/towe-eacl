import codecs
import numpy as np

from math import log
import scipy.sparse as sp

def load_text_target_label(path):
    text_list = []
    target_list = []
    label_list = []
    with codecs.open(path, encoding='utf-8') as fo:
        for i, line in enumerate(fo):
            if i == 0:
                continue
            s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
            text_list.append(sentence.strip())
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            target_list.append(target)
            w_l = opinion_words_tags.strip().split(' ')
            label = [l.split('\\')[-1] for l in w_l]
            label_list.append(label)
    return text_list, target_list, label_list


def split_dev(train_texts, train_t, train_ow):
    instances_index = []
    curr_s = ""
    curr_i = -1
    for i, s in enumerate(train_texts):
        s = ' '.join(s)

        if s == curr_s:
            instances_index[curr_i].append(i)
        else:
            curr_s = s
            instances_index.append([i])
            curr_i += 1
    # print(curr_i)
    # print(len(instances_index))
    assert curr_i + 1 == len(instances_index)
    length = len(instances_index)
    np.random.seed(1)
    index_list = np.random.permutation(length).tolist()
    # np.random.shuffle(index_list)
    train_index = [instances_index[i] for i in index_list[0:length - length // 5]]
    dev_index = [instances_index[i] for i in index_list[length - length // 5:]]
    train_i_index = [i for l in train_index for i in l]
    dev_i_index = [i for l in dev_index for i in l]
    dev_texts, dev_t, dev_ow = ([train_texts[i] for i in dev_i_index], [train_t[i] for i in dev_i_index],
                                [train_ow[i] for i in dev_i_index])
    train_texts, train_t, train_ow = ([train_texts[i] for i in train_i_index], [train_t[i] for i in train_i_index],
                                      [train_ow[i] for i in train_i_index])
    return train_texts, train_t, train_ow, dev_texts, dev_t, dev_ow, train_i_index, dev_i_index

def numericalize(text, vocab):
    if type(text)== str:
        tokens = text.split()
    else:
        tokens = text
    ids = []
    for token in tokens:
        token = token.lower()
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab['<UNK>'])
            # print('error:' + token) # stop warning
    assert len(ids) == len(tokens)
    return ids

def numericalize_label(labels, vocab):
    label_tensor = []
    for i, label in enumerate(labels):
        label_tensor.append(vocab[label])
    return label_tensor

def score_BIO(predicted, golden, ignore_index=-1):
    # O:0, B:1, I:2
    # print(predicted)
    assert len(predicted) == len(golden)
    sum_all = 0
    sum_correct = 0
    golden_01_count = 0
    predict_01_count = 0
    correct_01_count = 0
    # print(predicted)
    # print(golden)
    for i in range(len(golden)):
        length = len(golden[i])
        # print(length)
        # print(predicted[i])
        # print(golden[i])
        golden_01 = 0
        correct_01 = 0
        predict_01 = 0
        predict_items = []
        golden_items = []
        golden_seq = []
        predict_seq = []
        for j in range(length):
            if golden[i][j] == ignore_index:
                break
            if golden[i][j] == 1:
                if len(golden_seq) > 0:  # 00
                    golden_items.append(golden_seq)
                    golden_seq = []
                golden_seq.append(j)
            elif golden[i][j] == 2:
                if len(golden_seq) > 0:
                    golden_seq.append(j)
            elif golden[i][j] == 0:
                if len(golden_seq) > 0:
                    golden_items.append(golden_seq)
                    golden_seq = []
            if predicted[i][j] == 1:
                if len(predict_seq) > 0:  # 00
                    predict_items.append(predict_seq)
                    predict_seq = []
                predict_seq.append(j)
            elif predicted[i][j] == 2:
                if len(predict_seq) > 0:
                    predict_seq.append(j)
            elif predicted[i][j] == 0:
                if len(predict_seq) > 0:
                    predict_items.append(predict_seq)
                    predict_seq = []
        if len(golden_seq) > 0:
            golden_items.append(golden_seq)
        if len(predict_seq) > 0:
            predict_items.append(predict_seq)
        golden_01 = len(golden_items)
        predict_01 = len(predict_items)
        correct_01 = sum([item in golden_items for item in predict_items])
        # print(correct_01)
        # print([item in golden_items for item in predict_items])
        # print(golden_items)
        # print(predict_items)

        golden_01_count += golden_01
        predict_01_count += predict_01
        correct_01_count += correct_01
    precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
    recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
    return score_dict

def score_BIO_version_2(predicted, golden, ignore_index=-1):
    # O:1, B:2, I:3
    # print(predicted)
    assert len(predicted) == len(golden)
    sum_all = 0
    sum_correct = 0
    golden_01_count = 0
    predict_01_count = 0
    correct_01_count = 0
    # print(predicted)
    # print(golden)
    for i in range(len(golden)):
        length = len(golden[i])
        # print(length)
        # print(predicted[i])
        # print(golden[i])
        golden_01 = 0
        correct_01 = 0
        predict_01 = 0
        predict_items = []
        golden_items = []
        golden_seq = []
        predict_seq = []
        for j in range(length):
            if golden[i][j] == ignore_index:
                break
            if golden[i][j] == 2:
                if j == 0 or golden[i][j-1] == 2:
                    if len(golden_seq) > 0:  # 00
                        golden_items.append(golden_seq)
                        golden_seq = []
                    golden_seq.append(j)
                else:
                    if len(golden_seq) > 0:
                        golden_seq.append(j)
            elif golden[i][j] == 1:
                if len(golden_seq) > 0:
                    golden_items.append(golden_seq)
                    golden_seq = []

            if predicted[i][j] == 2:
                if j == 0 or predicted[i][j-1] == 2:
                    if len(predict_seq) > 0:  # 00
                        predict_items.append(predict_seq)
                        predict_seq = []
                    predict_seq.append(j)
                else:
                    if len(predict_seq) > 0:
                        predict_seq.append(j)

            elif predicted[i][j] == 1:
                if len(predict_seq) > 0:
                    predict_items.append(predict_seq)
                    predict_seq = []
        if len(golden_seq) > 0:
            golden_items.append(golden_seq)
        if len(predict_seq) > 0:
            predict_items.append(predict_seq)
        golden_01 = len(golden_items)
        predict_01 = len(predict_items)
        correct_01 = sum([item in golden_items for item in predict_items])
        # print(correct_01)
        # print([item in golden_items for item in predict_items])
        # print(golden_items)
        # print(predict_items)

        golden_01_count += golden_01
        predict_01_count += predict_01
        correct_01_count += correct_01
    precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
    recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
    return score_dict


def category_from_output(output):
    top_n, top_i = output.topk(1)  # Tensor out of Variable with .data
    # print(top_i)
    category_i = top_i.view(output.size()[0], -1).detach().cpu().numpy().tolist()
    return category_i