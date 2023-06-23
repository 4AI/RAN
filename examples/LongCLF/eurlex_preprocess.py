import os
import glob
import json

from tqdm import tqdm


base_dir = 'data/EURLEX57K'
data_dir = 'data/EURLEX57K/dataset'


def prepare_eurlex_data(eur_path, inverted=False):
    """
    Load EURLEX-57K dataset and prepare the datasets
    :param inverted: whether to invert the section order or not
    :param eur_path: path to the EURLEX files
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(eur_path):
        raise Exception("Data path not found: {}".format(eur_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}

    for split in ['train', 'dev', 'test']:
        file_paths = glob.glob(os.path.join(eur_path, split, '*.json'))
        for file_path in tqdm(sorted(file_paths)):
            text, tags = read_eurlex_file(file_path, inverted)
            text_set[split].append(text)
            label_set[split].append(tags)

    return text_set, label_set


def read_eurlex_file(eur_file_path, inverted):
    """
    Read each json file and return lists of documents and labels
    :param eur_file_path: path to a json file
    :param inverted: whether to invert the section order or not
    :return: list of documents and labels
    """
    tags = []
    with open(eur_file_path) as file:
        data = json.load(file)
    sections = []
    text = ''
    if inverted:
        sections.extend(data['main_body'])
        sections.append(data['recitals'])
        sections.append(data['header'])

    else:
        sections.append(data['header'])
        sections.append(data['recitals'])
        sections.extend(data['main_body'])

    text = '\n'.join(sections)

    for concept in data['concepts']:
        tags.append(concept)

    return text, tags


text_set, label_set = prepare_eurlex_data(data_dir)

for split in ['train', 'dev', 'test']:
    texts = text_set[split]
    labels = label_set[split]
    assert len(texts) == len(labels)
    print(f'{split} size:', len(texts))
    with open(os.path.join(base_dir, f'{split}.jsonl'), 'w') as writer:
        for text, label in zip(texts, labels):
            writer.writelines(json.dumps({'text': text, 'label': label}, ensure_ascii=False) + '\n')
