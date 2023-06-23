#! -*- coding:utf-8 -*-

import os
import re
import math
import json
from typing import List

# set TF_KERAS
os.environ['TF_KERAS'] = '1'  # noqa

import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from boltons.iterutils import chunked_iter
from langml import keras, L, K, TF_VERSION
from langml.baselines import BaseDataLoader
from langml.utils import pad_sequences
from rannet.rannet import RanNet
from rannet.tokenizer import RanNetWordPieceTokenizer


# set GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


num_classes = 2
batch_size = 16 # 25
eval_batch_size = 1
epochs = 50

# rannet
base_dir = 'rannet-base-v2-en-uncased-model'
config_path = os.path.join(base_dir, 'config.json')
checkpoint_path = os.path.join(base_dir, 'model.ckpt')
vocab_path = os.path.join(base_dir, 'vocab.txt')

# data path
train_path = 'data/hyperpartisan/train.jsonl'
valid_path = 'data/hyperpartisan/dev.jsonl'
test_path = 'data/hyperpartisan/test.jsonl'
save_model_path = 'ckpts/best_hp_model.weights'

# 建立分词器
min_length = None # 568
max_length = 4096
tokenizer = RanNetWordPieceTokenizer(vocab_path, lowercase=True)
tokenizer.enable_truncation(max_length=max_length)
label2idx = {'true': 1, 'false': 0}


def analyze_length(filepath):
    lengths = []
    with open(filepath, 'r') as reader:
        for line in reader:
            obj = json.loads(line)
            lengths.append(len(obj['text'].split()))
    df = pd.DataFrame({'length': lengths})
    print('length describe:')
    print(df['length'].describe())



def clean_text(string):
    string = string.replace("\n", "")
    string = string.replace("\t", "")

    string = re.sub(r'[0-9]+', '0', string)
    string = string.replace('ref_url=', '')
    string = re.sub(r'(/[a-zA-Z0-9\?=]+)+', '', string)
    string = re.sub(r'[A-Z0-9]+%', '', string)

    string = re.sub(r"[^A-Za-z0-9(),!?@\-\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class DataGenerator(BaseDataLoader):
    def __init__(self, tokenizer: RanNetWordPieceTokenizer, data: List, batch_size: int = 32):
        self.tokenizer = tokenizer
        self.data = data
        self.batch_size = batch_size

    @staticmethod
    def load_data(filename, is_test: bool = False):
        """ load data
        """
        data = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                obj = json.loads(l)
                text = clean_text(obj['text'])
                if min_length is not None and len(text.split()) < min_length:
                    text += ' | ' + ' '.join(text.split()[len(text.split()) - min_length:])
                data.append((text, label2idx[obj['label']]))
        return data

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def make_iter(self, random=False):
        if random:
            np.random.shuffle(self.data)

        for chunks in chunked_iter(self.data, self.batch_size):
            batch_tokens, batch_labels = [], []
            for text, label in chunks:
                tokenized = self.tokenizer.encode(text)
                batch_tokens.append(tokenized.ids)
                batch_labels.append([label])

            batch_tokens = pad_sequences(batch_tokens, padding='post', truncating='post')
            batch_labels = np.array(batch_labels)
            yield [batch_tokens], batch_labels


# build model
rannet, rannet_model = RanNet.load_rannet(
    config_path=config_path, checkpoint_path=checkpoint_path, return_sequences=False)
output = rannet_model.output
# output = L.Dense(128, activation='swish', kernel_initializer=rannet.initializer)(output)
output = L.Dropout(0.1)(output)
output = L.Dense(2, activation='softmax')(output)
model = keras.models.Model(rannet_model.input, output)
model.summary()

# analyze length
analyze_length(train_path)

train_data = DataGenerator.load_data(train_path)
valid_data = DataGenerator.load_data(valid_path)
test_data = DataGenerator.load_data(test_path)   # recover valid
print('train size:', len(train_data))
print('valid size:', len(valid_data))
train_generator = DataGenerator(tokenizer, train_data, batch_size)
valid_generator = DataGenerator(tokenizer, valid_data, eval_batch_size)
test_generator = DataGenerator(tokenizer, test_data, eval_batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true, verbose=False).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator.make_iter())
        test_acc = evaluate(test_generator.make_iter())
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(save_model_path)
        print(
            'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':
    evaluator = Evaluator()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(5e-5),
        metrics=['accuracy'],
    )
    model.fit(
        train_generator(random=True),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights(save_model_path)
    for i, (text, label) in enumerate(test_data):
        tok = tokenizer.encode(text)
        logits = model.predict([np.array([tok.ids])])[0]
        pred = logits.argmax()
        if label != pred:
            print('wrong:')
            print('idx:', i)
            print('text:', text)
            print('pred:', pred)
            print('true:', label)
            print()
else:
    model.load_weights(save_model_path)
    while True:
        text = input('>> ')
        tok = tokenizer.encode(text)
        logits = model.predict([np.array([tok.ids])])[0]
        pred = logits.argmax()
        print('pred:', pred)