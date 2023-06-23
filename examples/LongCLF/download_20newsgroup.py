# -*- coding: utf-8 -*-

import os
import re
import json

from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


save_dir = 'data/20newsgroup'
os.makedirs(save_dir, exist_ok=True)


def clean_20news_data(text_str):
    """
    Clean up 20NewsGroups text data, from CogLTX: https://github.com/Sleepychord/CogLTX/blob/main/20news/process_20news.py
    // SPDX-License-Identifier: MIT
    :param text_str: text string to clean up
    :return: clean text string
    """
    tmp_doc = []
    for words in text_str.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc


def prepare_20news_data():
    """
    Load the 20NewsGroups datasets and split the original train set into train/dev sets
    :return: dicts of lists of documents and labels and number of labels
    """
    text_set = {}
    label_set = {}
    test_set = fetch_20newsgroups(subset='test', random_state=21)
    text_set['test'] = [clean_20news_data(text) for text in test_set.data]
    label_set['test'] = test_set.target

    train_set = fetch_20newsgroups(subset='train', random_state=21)
    text_set['train'] = [clean_20news_data(text) for text in train_set.data]
    label_set['train'] = train_set.target

    return text_set, label_set


text_set, label_set = prepare_20news_data()

for split in ['train', 'test']:
    texts = text_set[split]
    labels = label_set[split]
    assert len(texts) == len(labels)
    print(f'{split} size:', len(texts))
    with open(os.path.join(save_dir, f'{split}.jsonl'), 'w') as writer:
        for text, label in zip(texts, labels):
            writer.writelines(json.dumps({'text': text, 'label': str(label)}, ensure_ascii=False) + '\n')
