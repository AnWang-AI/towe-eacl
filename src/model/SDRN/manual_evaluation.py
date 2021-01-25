import os
import argparse

def get_prediction_result_from_absa_result_file(absa_result_file):
    with open(absa_result_file, encoding='utf-8') as f:
        raw_text = f.read().strip()

    absa_label_map = {'O': 0, 'B-T': 0, 'I-T': 0, 'B-P': 1, 'I-P': 2}

    cases = []
    samples = raw_text.split('\n\n')
    for sample in samples:
        case_cache, grounth_truth_case_cache = {}, {}
        sentence, golden_labels, predicted_labels = [], [], []
        begin_relation_flag = False
        relations = []
        for line in sample.split('\n'):
            if line.strip() == '#Relations':
                begin_relation_flag = True
                continue

            if begin_relation_flag:
                relations.append([int(v) for v in line.strip().split('\t')])
            else:
                word, golden_label, predicted_label = line.strip().split('\t')
                sentence.append(word)
                golden_labels.append(golden_label)
                predicted_labels.append(predicted_label)

        for relation in relations:
            target_key = ''.join([string.replace('#', '') for string in sentence[relation[2]: relation[3]]])
            if target_key not in case_cache.keys():
                prediction_result = [0] * len(sentence)
                for k in range(relation[0], relation[1]):
                    prediction_result[k] = absa_label_map[predicted_labels[k]]
                prediction_result[relation[0]] = absa_label_map[predicted_labels[relation[0]]]
                case_cache[target_key] = prediction_result
            else:
                for k in range(relation[0], relation[1]):
                    case_cache[target_key][k] = absa_label_map[predicted_labels[k]]
                case_cache[target_key][relation[0]] = absa_label_map[predicted_labels[relation[0]]]
        cases.append(case_cache)
    return cases


def get_grounth_truth_from_absa_file(absa_file):
    with open(absa_file, encoding='utf-8') as f:
        raw_text = f.read().strip()

    cases = []
    samples = raw_text.split('\n\n')
    for sample in samples:
        begin_relation_flag = False
        relations = []
        sentence, golden_labels = [], []
        case_cache = {}
        for line in sample.split('\n'):
            if line.strip() == '#Relations':
                begin_relation_flag = True
                continue

            if begin_relation_flag:
                relations.append([int(v) for v in line.strip().split('\t')])
            else:
                word, golden_label = line.strip().split('\t')
                sentence.append(word)
                golden_labels.append(golden_label)

            for relation in relations:
                target_key = ''.join([string.replace('#', '') for string in sentence[relation[2]: relation[3]]])
                if target_key not in case_cache.keys():
                    prediction_result = [0] * len(sentence)
                    for k in range(relation[0], relation[1]):
                        prediction_result[k] = 2
                    prediction_result[relation[0]] = 1
                    case_cache[target_key] = prediction_result
                else:
                    for k in range(relation[0], relation[1]):
                        case_cache[target_key][k] = 2
                    case_cache[target_key][relation[0]] = 1
        cases.append(case_cache)
    return cases


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


def manual_evaluate(prediction_results, grounth_truths):
    pred_list = []
    label_list = []
    for prediction_result, grounth_truth in zip(prediction_results, grounth_truths):
        for key, value in grounth_truth.items():
            label_list.append(value)
            if key in prediction_result:
                pred_list.append(prediction_result[key])
            else:
                pred_list.append([0] * len(value))
    score_dict = score_BIO(pred_list, label_list)
    BIO_score = score_dict["f1"]
    BIO_info = 'BIO precision: {:.4f}, BIO recall: {:.4f}, BIO f1: {:.4f}'.format(score_dict["precision"],
                                                                                  score_dict["recall"],
                                                                                  score_dict["f1"])
    # print(BIO_info)
    return score_dict["precision"], score_dict["recall"], score_dict["f1"]


def manual_evaluate_by_file(test_output_file, absa_file):
    prediction_results = get_prediction_result_from_absa_result_file(test_output_file)
    grounth_truths = get_grounth_truth_from_absa_file(absa_file)
    p, r, f1 = manual_evaluate(prediction_results, grounth_truths)
    return p, r, f1


def manual_evaluate_by_files(test_output_dir, absa_file, master_metric='f1', select_top=1):
    metrics = []
    filenames = os.listdir(test_output_dir)
    for filename in filenames:
        p, r, f1 = manual_evaluate_by_file(os.path.join(test_output_dir, filename), absa_file)
        metrics.append([p, r, f1])

    if master_metric == 'p':
        metrics = sorted(metrics, key=lambda x: x[0], reverse=True)
    elif master_metric == 'r':
        metrics = sorted(metrics, key=lambda x: x[1], reverse=True)
    elif master_metric == 'f1':
        metrics = sorted(metrics, key=lambda x: x[2], reverse=True)

    for i in range(select_top):
        BIO_info = 'BIO precision: {:.4f}, BIO recall: {:.4f}, BIO f1: {:.4f}'.format(metrics[i][0],
                                                                                      metrics[i][1],
                                                                                      metrics[i][2])
        print('top %s:' % (i + 1) + BIO_info)


def manual_evaluate_by_dataset(dataset_name):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_output_dir', type=str)
    parser.add_argument('--absa_file', type=str)
    args = parser.parse_args()

    manual_evaluate_by_files(test_output_dir=args.test_output_dir,
                             absa_file=args.absa_file,
                             master_metric='f1',
                             select_top=1)
