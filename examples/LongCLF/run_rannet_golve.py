# -*- coding: utf-8 -*-

'''

# agnews: python run_rannet_glove.py --train_path data/agnews/train.jsonl --test_path data/agnews/test.jsonl --max_sequence_len 128 --batch_size 32 --epochs 50 --learning_rate 0.0005

# 20ng: python run_rannet_glove.py --train_path data/20newsgroup/train.jsonl --test_path data/20newsgroup/test.jsonl --max_sequence_len 1024 --batch_size 32 --epochs 50 --learning_rate 0.0005

# booksummary: python run_rannet_glove.py --train_path data/booksummaries/train.jsonl --test_path data/booksummaries/test.jsonl --max_sequence_len 1024 --batch_size 16 --epochs 50 --learning_rate 0.0001 --dropout_rate 0.45

# hyperpartisan: python run_rannet_glove.py --train_path data/hyperpartisan/train.jsonl --test_path data/hyperpartisan/test.jsonl --max_sequence_len 2048 --learning_rate 0.0005 --batch_size 16 --epochs 50

# eurlex57k: python run_rannet_glove.py --train_path data/EURLEX57K/train.jsonl --test_path data/EURLEX57K/test.jsonl --max_sequence_len 2048 --learning_rate 0.0005 --batch_size 16 --epochs 50

# arxiv: CUDA_VISIBLE_DEVICES=4 python run_rannet_glove.py --train_path data/arxiv-clf/train.jsonl --test_path data/arxiv-clf/test.jsonl --max_sequence_len 2048 --learning_rate 0.0005 --batch_size 16 --epochs 50

GPU Server

multi layers, cell initialization with default:
# arxiv: CUDA_VISIBLE_DEVICES=7 python run_rannet_glove.py --glove_path pretrained-embedding/glove.6B.300d.txt --train_path data/arxiv-clf/train.jsonl --test_path data/arxiv-clf/test.jsonl --max_sequence_len 2048 --learning_rate 0.0005 --batch_size 16 --epochs 50 --n_layers 1
# arxiv: CUDA_VISIBLE_DEVICES=1 python run_rannet_glove.py --glove_path pretrained-embedding/glove.6B.300d.txt --train_path data/arxiv-clf/train.jsonl --test_path data/arxiv-clf/test.jsonl --max_sequence_len 2048 --learning_rate 0.0005 --batch_size 16 --epochs 50 --n_layers 2
# arxiv: CUDA_VISIBLE_DEVICES=2 python run_rannet_glove.py --glove_path pretrained-embedding/glove.6B.300d.txt --train_path data/arxiv-clf/train.jsonl --test_path data/arxiv-clf/test.jsonl --max_sequence_len 2048 --learning_rate 0.0002 --batch_size 8 --epochs 50 --n_layers 3 --ran_dropout_rate 0.2
# arxiv: CUDA_VISIBLE_DEVICES=2 python run_rannet_glove.py --glove_path pretrained-embedding/glove.6B.300d.txt --train_path data/arxiv-clf/train.jsonl --test_path data/arxiv-clf/test.jsonl --max_sequence_len 2048 --learning_rate 0.00025 --batch_size 8 --epochs 50 --n_layers 4 --ran_dropout_rate 0.2


different window size:

titan: CUDA_VISIBLE_DEVICES=0 python run_rannet_glove.py --train_path data/arxiv-clf/train.jsonl --test_path data/arxiv-clf/test.jsonl --max_sequence_len 2048 --learning_rate 0.0005 --batch_size 8 --epochs 50 --n_layers 2 --window_size 1024
gpuserver: CUDA_VISIBLE_DEVICES=6 python run_rannet_glove.py --glove_path pretrained-embedding/glove.6B.300d.txt --train_path data/arxiv-clf/train.jsonl --test_path data/arxiv-clf/test.jsonl --max_sequence_len 2048 --learning_rate 0.0005 --batch_size 16 --epochs 50 --n_layers 2 --window_size 512

'''


import argparse
import json
from typing import List, Optional

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Embedding, Lambda, Concatenate
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
try:
    from keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras.utils import pad_sequences
from sklearn.metrics import f1_score, accuracy_score
from langml.layers import LayerNorm
from rannet.ran import RAN, GatedLinearUnit


# set GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



def load_data(fpath, label2idx=None):
    texts = []
    labels = []
    multi_label = False
    build_vocab = False
    if label2idx is None:
        build_vocab = True
        label2idx = {}

    with open(fpath, 'r') as reader:
        for line in reader:
            obj = json.loads(line)
            texts.append(obj['text'])
            if isinstance(obj['label'], list):
                multi_label = True

            if build_vocab:
                current_labels = [obj['label']] if not isinstance(obj['label'], list) else obj['label']
                for label in current_labels:
                    if label not in label2idx:
                        label2idx[label] = len(label2idx)
            labels.append([label2idx[v] for v in obj['label'] if v in label2idx] if isinstance(obj['label'], list) else label2idx[obj['label']])
    if multi_label:
        one_hot_labels = []
        for label_list in labels:
            label_vector = np.zeros(len(label2idx))
            for label in label_list:
                label_vector[label] = 1
            one_hot_labels.append(label_vector)
        return texts, one_hot_labels, label2idx, multi_label
    return texts, labels, label2idx, multi_label


def evaluate_accuracy(model, data, labels):
    y_pred = model.predict(data, verbose=False).argmax(axis=1)
    y_true = labels.argmax(axis=1)
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def evaluate_micro_f1(model, data, labels):
    y_pred = model.predict(data, verbose=False)
    y_pred = (y_pred > 0.5).astype('int')
    y_true = labels
    # print('y true:', y_true, y_true.shape)
    # print('y pred:', y_pred, y_pred.shape)
    return f1_score(y_true, y_pred, average='micro')


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self, model, data, labels, is_multi_label):
        self.model = model
        self.data = data
        self.labels = labels
        self.evaluate = evaluate_micro_f1 if is_multi_label else evaluate_accuracy
        self.metric_name = 'micro f1' if is_multi_label else 'accuracy'
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = self.evaluate(self.model, self.data, self.labels)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
        
        print(
            f'val {self.metric_name}: %.5f, best val {self.metric_name}: %.5f\n' %
            (val_score, self.best_val_score)
        )


def main(config):

    print('Indexing word vectors.')
    embeddings_index = {}
    with open(config.glove_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))

    train_texts, train_labels, label2idx, is_multi_label = load_data(config.train_path)
    label_size = len(label2idx)
    test_texts, test_labels, _, _ = load_data(config.test_path, label2idx=label2idx)
    print(f'is_multi_label={is_multi_label}')
    print(f'label size: {label_size}')

    num_words = 40000 if 'arxiv' in config.train_path.lower() else None
    print(f'num_words of tokenizer = {num_words}')
    tokenizer = Tokenizer(num_words=num_words, lower=True)
    tokenizer.fit_on_texts(train_texts)
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    # for test
    # train_texts, train_labels = train_texts[:1000], train_labels[:1000]
    # test_texts, test_labels = test_texts[:200], test_labels[:200]

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    x_train = pad_sequences(train_sequences, maxlen=config.max_sequence_len, padding='post', truncating='post')
    x_test = pad_sequences(test_sequences, maxlen=config.max_sequence_len, padding='post', truncating='post')
    if is_multi_label:
        y_train = np.asarray(train_labels)
        y_test = np.asarray(test_labels)
    else:
        y_train = to_categorical(np.asarray(train_labels))
        y_test = to_categorical(np.asarray(test_labels))
    print('Shape of train data tensor:', x_train.shape, ', train label tensor:', y_train.shape)
    print('Shape of test data tensor:', x_test.shape, ', test label tensor:', y_test.shape)

    # Prepare the embedding matrix:
    print('Preparing embedding matrix.')

    num_words = len(word_index) + 1 if num_words is None else num_words
    embedding_dim = config.glove_embed_dim

    oov = 0
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix = np.random.uniform(-0.2, 0.2, (num_words, embedding_dim))
    for word, i in word_index.items():
        if num_words is not None and i >= num_words:
            # https://github.com/keras-team/keras/blob/v2.11.0/keras/preprocessing/text.py#L398
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            oov += 1
    print('OOV size:', oov)
    print('Shape of embedding matrix:', embedding_matrix.shape)

    # ### Initialization

    print('Build model...')
    x_in = Input(shape=(None, ), name='Input-Token')
    mask = Lambda(lambda t: K.cast(K.greater(t, 0), K.floatx()), name='Text-Mask')(x_in)
    x = x_in
    x = Embedding(num_words,
                  embedding_dim,
                  weights=[embedding_matrix],
                  mask_zero=True,
                  trainable=True)(x)
    x = Dropout(config.dropout_rate)(x)
    # x = LayerNorm()(x)
    x, cell = RAN(8, 128, window_size=config.window_size, dropout_rate=config.ran_dropout_rate, min_window_size=config.min_window_size)([x, mask])
    for _ in range(config.n_layers - 1):
        x_t, cell = RAN(
            8, 128,
            window_size=config.window_size,
            dropout_rate=config.ran_dropout_rate,
            min_window_size=config.min_window_size)([x, mask], cell=cell if config.cell_initialization == 'default' else None)
        if config.n_layers > 2:
            # residual for deep layers (layers > 2)
            x = Lambda(lambda x: x[0] + x[1])([x, x_t])
        else:
            x = x_t
    if config.pooling_strategy == 'avg':
        # avg pool
        pool = Lambda(lambda x: K.mean(x, axis=1))(x)
    else:
        # max pool
        pool = Lambda(lambda x: K.max(x, axis=1))(x)
    fea_dim = K.int_shape(pool)[-1]
    if config.fusion_strategy == 'concat':
        o = Concatenate()([
            Dense(fea_dim, activation='swish')(pool),
            Dense(fea_dim, activation='swish')(cell),
        ])
    else:
        # add
        pool = Dense(fea_dim, activation='swish')(pool)
        o = Lambda(lambda x: x[0] + x[1])([pool, GatedLinearUnit(fea_dim)(cell)]) # pool + glu(cell)
    o = Dense(2 * label_size if is_multi_label else 128, activation='swish')(o)
    x = Dense(label_size, activation='sigmoid' if is_multi_label else 'softmax')(o)
    model = keras.Model(x_in, x)
    model.summary()
    model.compile(loss='binary_crossentropy' if is_multi_label else 'categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(config.learning_rate),
                  metrics=['categorical_accuracy' if is_multi_label else 'accuracy'])

    model.fit(x_train, y_train,
              batch_size=config.batch_size,
              epochs=config.epochs,
              validation_data=(x_test, y_test),
              verbose=1,
              callbacks=[Evaluator(model, x_test, y_test, is_multi_label)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--min_window_size', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--ran_dropout_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--cell_initialization', type=str, default='default')  # default, zeros
    parser.add_argument('--glove_path', type=str, default='./glove.6B.300d.txt')
    parser.add_argument('--glove_embed_dim', type=int, default=300)
    parser.add_argument('--train_path', type=str, default='train.jsonl')
    parser.add_argument('--test_path', type=str, default='test.jsonl')
    parser.add_argument('--max_sequence_len', type=int, default=512)
    parser.add_argument('--pooling_strategy', type=str, default='max')
    parser.add_argument('--fusion_strategy', type=str, default='add')
    config = parser.parse_args()
    main(config)
