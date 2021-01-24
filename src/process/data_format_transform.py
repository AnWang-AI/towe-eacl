import argparse


def towe_data_to_absa_data_strict(towe_file, absa_file):
    """
        将towe的数据格式转换为absa的数据格式
        并且将同id的合并
    """
    with open(absa_file, 'w', encoding='utf-8') as writer:
        pass

    with open(towe_file, encoding='utf-8') as f:
        header = f.readline()
        pre_s_id = "-1"
        for i in f:
            s_id, sentence, target_tags, opinion_words_tags = i.strip().split('\t')
            if s_id != pre_s_id:
                if pre_s_id != '-1':
                    with open(absa_file, 'a', encoding='utf-8') as writer:
                        sample = ''
                        for index, word in enumerate(pre_sentence.strip().split(' ')):
                            sample += '%s\t%s\n' % (word, pre_label_map[index])
                        sample += '#Relations\n'
                        for relation in pre_relations:
                            sample += '\t'.join(relation) + '\n'
                        sample += '\n'
                        writer.write(sample)
                label_map = []
                relations = []
                for index, word in enumerate(sentence.strip().split(' ')):
                    label_map.append('O')

            pre_s_id = s_id
            pre_sentence = sentence

            target_start, target_end = -1, -1
            for index, word in enumerate(target_tags.strip().split(' ')):
                label = word[-1]
                if label == 'B':
                    label_map[index] = 'B-T'
                    target_start = index
                    target_end = index
                elif label == 'I':
                    label_map[index] = 'I-T'
                    target_end = index

            opinion_start, opinion_end = [], []
            num_opinion = -1
            new_opinion_flag = True
            for index, word in enumerate(opinion_words_tags.strip().split(' ')):
                label = word[-1]
                if label == 'B':
                    label_map[index] = 'B-P'
                    if new_opinion_flag:
                        num_opinion += 1
                        opinion_start.append(-1)
                        opinion_end.append(-1)
                    opinion_start[num_opinion] = index
                    opinion_end[num_opinion] = index
                    new_opinion_flag = True
                elif label == 'I':
                    label_map[index] = 'I-P'
                    opinion_end[num_opinion] = index

            for start, end in zip(opinion_start, opinion_end):
                relations.append([str(v) for v in [start, end + 1, target_start, target_end + 1]])

            pre_label_map = label_map
            pre_relations = relations

    # last case
    with open(absa_file, 'a', encoding='utf-8') as writer:
        sample = ''
        for index, word in enumerate(pre_sentence.strip().split(' ')):
            sample += '%s\t%s\n' % (word, pre_label_map[index])
        sample += '#Relations\n'
        for relation in pre_relations:
            sample += '\t'.join(relation) + '\n'
        sample += '\n'
        writer.write(sample)


def towe_data_to_absa_data(towe_file, absa_file):
    """
        将towe的数据格式转换为absa的数据格式
    """
    with open(absa_file, 'w', encoding='utf-8') as writer:
        pass

    with open(towe_file, encoding='utf-8') as f:
        header = f.readline()
        for i in f:
            s_id, sentence, target_tags, opinion_words_tags = i.strip().split('\t')
            label_map = []
            for index, word in enumerate(sentence.strip().split(' ')):
                label_map.append('O')

            target_start, target_end = -1, -1
            for index, word in enumerate(target_tags.strip().split(' ')):
                label = word[-1]
                if label == 'B':
                    label_map[index] = 'B-T'
                    target_start = index
                    target_end = index
                elif label == 'I':
                    label_map[index] = 'I-T'
                    target_end = index

            opinion_start, opinion_end = [], []
            num_opinion = -1
            new_opinion_flag = True
            for index, word in enumerate(opinion_words_tags.strip().split(' ')):
                label = word[-1]
                if label == 'B':
                    label_map[index] = 'B-P'
                    if new_opinion_flag:
                        num_opinion += 1
                        opinion_start.append(-1)
                        opinion_end.append(-1)
                    opinion_start[num_opinion] = index
                    opinion_end[num_opinion] = index
                    new_opinion_flag = True
                elif label == 'I':
                    label_map[index] = 'I-P'
                    opinion_end[num_opinion] = index

            with open(absa_file, 'a', encoding='utf-8') as writer:
                sample = ''
                for index, word in enumerate(sentence.strip().split(' ')):
                    sample += '%s\t%s\n' % (word, label_map[index])
                sample += '#Relations\n'
                for start, end in zip(opinion_start, opinion_end):
                    sample += '%s\t%s\t%s\t%s\n' % (start, end + 1, target_start, target_end + 1)
                sample += '\n'
                writer.write(sample)


def absa_data_to_towe_data(absa_file, towe_file):
    """
        将absa的数据格式转换为towe的数据格式
    """
    with open(towe_file, 'w', encoding='utf-8') as writer:
        header = 's_id\tsentence\ttarget_tags\topinion_words_tags\n'
        writer.write(header)

    with open(absa_file, encoding='utf-8') as f:
        raw_text = f.read().strip()

    samples = raw_text.split('\n\n')
    for index, sample in enumerate(samples):
        s_id = index
        sentence = []
        label_map = {}
        for line in sample.split('\n'):
            if line.strip() == '#Relations':
                break
            word, label = line.strip().split('\t')
            label_map[word] = label
            sentence.append(word)

        target_tags, opinion_words_tags = [], []
        exist_target, exist_opinion = False, False
        target_count = 0
        for word in sentence:
            if label_map[word] == 'O':
                target_tags.append(word + r'\O')
                opinion_words_tags.append(word + r'\O')
            elif label_map[word] == 'B-T':
                target_tags.append(word + r'\B')
                exist_target = True
                target_count += 1
                opinion_words_tags.append(word + r'\O')
            elif label_map[word] == 'I-T':
                target_tags.append(word + r'\I')
                exist_target = True
                opinion_words_tags.append(word + r'\O')
            elif label_map[word] == 'B-P':
                target_tags.append(word + r'\O')
                opinion_words_tags.append(word + r'\B')
                exist_opinion = True
            elif label_map[word] == 'I-P':
                target_tags.append(word + r'\O')
                opinion_words_tags.append(word + r'\I')
                exist_opinion = True

        if exist_target and exist_opinion and target_count == 1:
            with open(towe_file, 'a', encoding='utf-8') as writer:
                writer.write('%s\t%s\t%s\t%s\n' % (s_id, ' '.join(sentence), ' '.join(target_tags), ' '.join(opinion_words_tags)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--towe_file', type=str, required=True)
    parser.add_argument('--absa_file', type=str, required=True)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    if args.mode == 'towe2absa':
        if args.strict:
            towe_data_to_absa_data_strict(towe_file=args.towe_file, absa_file=args.absa_file)
        else:
            towe_data_to_absa_data(towe_file=args.towe_file, absa_file=args.absa_file)
    elif args.mode == 'absa2towe':
        absa_data_to_towe_data(absa_file=args.absa_file, towe_file=args.towe_file)
