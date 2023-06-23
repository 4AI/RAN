# -*- coding: utf-8 -*-

import os
import json

from tqdm import tqdm
from datasets import load_dataset



save_dir = 'data/agnews'
os.makedirs(save_dir, exist_ok=True)

dataset = load_dataset("ag_news")

print('save train set')
with open(os.path.join(save_dir, 'train.jsonl'), 'w') as writer:
    for obj in tqdm(dataset['train']):
        writer.writelines(json.dumps({'text': obj['text'], 'label': obj['label']}, ensure_ascii=False) + '\n')


print('save test set')
with open(os.path.join(save_dir, 'test.jsonl'), 'w') as writer:
    for obj in tqdm(dataset['test']):
        writer.writelines(json.dumps({'text': obj['text'], 'label': obj['label']}, ensure_ascii=False) + '\n')
