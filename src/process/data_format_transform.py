import argparse


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
            label_map = {}
            for word in sentence.strip().split(' '):
                label_map[word] = 'O'
            for word in target_tags.strip().split(' '):
                label = word[-1]
                if label == 'B':
                    label_map[word[:-2]] = 'B-T'
                elif label == 'I':
                    label_map[word[:-2]] = 'I-T'
            for word in opinion_words_tags.strip().split(' '):
                label = word[-1]
                if label == 'B':
                    label_map[word[:-2]] = 'B-P'
                elif label == 'I':
                    label_map[word[:-2]] = 'I-P'
            with open(absa_file, 'a', encoding='utf-8') as writer:
                sample = ''
                for word in sentence.strip().split(' '):
                    sample += '%s\t%s\n' % (word, label_map[word])
                sample += '#Relations\n\n'
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
    parser.add_argument('--towe_file', type=str, required=True)
    parser.add_argument('--absa_file', type=str, required=True)
    args = parser.parse_args()

    # towe_data_to_absa_data(towe_file=args.towe_file,
    #                        absa_file=args.absa_file)
    absa_data_to_towe_data(absa_file=args.absa_file,
                           towe_file=args.towe_file)
