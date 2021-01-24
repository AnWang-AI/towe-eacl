def extract_target_opinion_pair_from_result(absa_result_file):
    with open(absa_result_file, encoding='utf-8') as f:
        raw_text = f.read().strip()

    label_map = {'O': 0, 'B': 1, 'I': 2}

    samples = raw_text.split('\n\n')
    all_target_indices, all_opinion_indices = [], []
    for index, sample in enumerate(samples):
        s_id = index
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

        target_indices, opinion_indices = [], []

        pre_target_index = []
        opinion_cache = []
        for relation in relations:
            current_target_index = [int(relation[2]), int(relation[3])]

            if current_target_index != pre_target_index and opinion_cache != []:
                target_indices.append(pre_target_index)
                opinion_indices.append(opinion_cache)
                opinion_cache = []

            opinion_cache.append([int(relation[0]), int(relation[1])])
            pre_target_index = current_target_index

        if opinion_cache != []:
            target_indices.append(pre_target_index)
            opinion_indices.append(opinion_cache)

        all_target_indices.append(target_indices)
        all_opinion_indices.append(opinion_indices)
    return all_target_indices, all_opinion_indices


def make_prediction_label_from_absa_result(absa_result_file):
    all_target_indices, all_opinion_indices = extract_target_opinion_pair_from_result(absa_result_file)
    with open(absa_result_file, encoding='utf-8') as f:
        raw_text = f.read().strip()

    absa_label_map = {'O': 0, 'B-T': 0, 'I-T': 0, 'B-P': 1, 'I-P': 2}

    samples = raw_text.split('\n\n')
    for sample, target_indices, opinion_indices in zip(samples, all_target_indices, all_opinion_indices):
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

        for target_indice in target_indices:
            print(target_indice)


def get_grounth_truth_relations_from_absa_file(absa_file):
    with open(absa_file, encoding='utf-8') as f:
        raw_text = f.read().strip()

    all_relations = []
    samples = raw_text.split('\n\n')
    for sample in samples:
        begin_relation_flag = False
        relations = []
        for line in sample.split('\n'):
            if line.strip() == '#Relations':
                begin_relation_flag = True
                continue

            if begin_relation_flag:
                relations.append([int(v) for v in line.strip().split('\t')])
        all_relations.append(relations)
    return all_relations


def get_prediction_result_from_absa_result_file(absa_result_file):
    with open(absa_result_file, encoding='utf-8') as f:
        raw_text = f.read().strip()

    absa_label_map = {'O': 0, 'B-T': 0, 'I-T': 0, 'B-P': 1, 'I-P': 2}

    cases = []
    samples = raw_text.split('\n\n')
    for sample in samples:
        case_cache = {}
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
            if '%s-%s' % (relation[2], relation[3]) not in case_cache.keys():
                prediction_result = [0] * len(sentence)
                for k in range(relation[0], relation[1]):
                    prediction_result[k] = 2
                prediction_result[relation[0]] = 1
                case_cache['%s-%s' % (relation[2], relation[3])] = prediction_result
            else:
                for k in range(relation[0], relation[1]):
                    case_cache['%s-%s' % (relation[2], relation[3])][k] = 2
                case_cache['%s-%s' % (relation[2], relation[3])][relation[0]] = 1
        cases.append(case_cache)
    return cases


def get_grounth_truth_from_towe_file(towe_file):
    towe_label_map = {'O': 0, 'B': 1, 'I': 2}
    with open(towe_file) as f:
        header = f.readline()
        lines = f.readlines()

    cases = []
    pre_s_id = '-1'
    case_cache = {}
    for line in lines:
        s_id, sentence, target_tags, opinion_words_tags = line.strip().split('\t')
        if s_id != pre_s_id and case_cache != {}:
            cases.append(case_cache)
            case_cache = {}

        target_index = [0, 0]
        for i, item in enumerate(target_tags.split(' ')):
            tag = item[-1]
            if tag == 'B':
                target_index[0] = i
                target_index[1] = i + 1
            elif tag == 'I':
                target_index[1] = i + 1

        golden_labels = []
        for item in opinion_words_tags.split(' '):
            tag = item[-1]
            golden_labels.append(towe_label_map[tag])
        case_cache['%s-%s' % tuple(target_index)] = golden_labels

        pre_s_id = s_id
    if case_cache != {}:
        cases.append(case_cache)

    return cases


def get_case_number_from_towe_file(towe_file):
    # 从towe file里看每一个id有多少个case
    with open(towe_file) as f:
        header = f.readline()
        lines = f.readlines()

    pre_s_id = '-1'
    counter = 0
    case_number = []
    s_ids = []
    for line in lines:
        s_id, _, _, _ = line.strip().split('\t')
        if s_id != pre_s_id and counter > 0:
            case_number.append(counter)
            s_ids.append(pre_s_id)
            counter = 0
        pre_s_id = s_id
        counter += 1
    if counter > 0:
        case_number.append(counter)
    return case_number, s_ids


if __name__ == '__main__':
    towe_file = '/Users/jiangjunfeng/mainland/private/towe-eacl/data/14res/test.tsv'
    # absa_file = '/Users/jiangjunfeng/mainland/private/towe-eacl/data/SDRN/14res.test'
    absa_result_file = '/Users/jiangjunfeng/mainland/private/towe-eacl/data/SDRN/test_output_69'
    # all_target_indices, all_opinion_indices = extract_target_opinion_pair_from_result(absa_result_file)
    # print(len(all_target_indices), len(all_opinion_indices))
    #
    # towe_file = '/Users/jiangjunfeng/mainland/private/towe-eacl/data/14res/test.tsv'
    # case_numbers, s_ids = get_case_number_from_towe_file(towe_file)
    # print(len(case_numbers))
    #
    # matched_num = 0
    # for target_indices, case_number, s_id in zip(all_target_indices, case_numbers, s_ids):
    #     if len(target_indices) == case_number:
    #         matched_num += 1
    #
    #     print("%s -- %s -- sid: %s" % (len(target_indices), case_number, s_id))
    #
    # print(matched_num / len(case_numbers))

    # make_prediction_label_from_absa_result(absa_result_file)
    towe_cases = get_grounth_truth_from_towe_file(towe_file)
    print(towe_cases[:5])

    absa_cases = get_prediction_result_from_absa_result_file(absa_result_file)
    print(absa_cases[:5])
