# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datasets import load_dataset
from scipy.stats import spearmanr
from langml import keras, K, L
from langml.utils import pad_sequences

from rannet import RanNet, RanNetWordPieceTokenizer


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compute_kernel_bias(vecs):
    """ modified from: https://github.com/bojone/BERT-whitening
    计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """ modified from: https://github.com/bojone/BERT-whitening
    应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def l2_normalize(vecs):
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def convert2ids(tokenizer, dataset):
    a_token_ids = []
    b_token_ids = []
    labels = []
    for obj in tqdm(dataset):
        a_token_ids.append(tokenizer.encode(obj['sentence1']).ids)
        b_token_ids.append(tokenizer.encode(obj['sentence2']).ids)
        labels.append(int(obj['label']))
    a_token_ids = pad_sequences(a_token_ids, truncating='post', padding='post')
    b_token_ids = pad_sequences(b_token_ids, truncating='post', padding='post')
    return a_token_ids, b_token_ids, labels


def evaluate(base_dir, dataset, name,
             min_window_size=16,
             window_size=32,
             strategy='first-last-avgpool-avg',
             apply_whitening=True,
             n_components=64,
             n_samples=None):
    tokenizer = RanNetWordPieceTokenizer(f'{base_dir}/vocab.txt', lowercase=True)
    model = build_model(base_dir, min_window_size=min_window_size, window_size=window_size, strategy=strategy)
    for split in ['test']: # ['train', 'validation', 'test']:
        print(f'evaluate {name}-{split}!')
        a_token_ids, b_token_ids, labels = convert2ids(tokenizer, dataset[split])
        if n_samples is not None:
            a_token_ids = a_token_ids[:n_samples, :]
            b_token_ids = b_token_ids[:n_samples, :]
            labels = labels[:n_samples]
        print('a_token_ids shape:', a_token_ids.shape)
        print('b_token_ids shape:', b_token_ids.shape)

        a_vecs = model.predict(a_token_ids, verbose=True)
        b_vecs = model.predict(b_token_ids, verbose=True)

        print('a_vecs shape:', a_vecs.shape)
        print('b_vecs shape:', b_vecs.shape)
        print('a vecs:', a_vecs)
        print('b vecs:', b_vecs)
        # whitening
        if apply_whitening:
            all_vecs = [(a_vecs, b_vecs)]
            kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])
            print('kernel shape:', kernel.shape)
            kernel = kernel[:, :n_components]
            a_vecs = transform_and_normalize(a_vecs, kernel, bias)
            b_vecs = transform_and_normalize(b_vecs, kernel, bias)
        else:
            a_vecs = l2_normalize(a_vecs)
            b_vecs = l2_normalize(b_vecs)
        print('a vecs:', a_vecs)
        print('b vecs:', b_vecs)
        sims = (a_vecs * b_vecs).sum(axis=1)
        corrcoef = spearmanr(labels, sims).correlation

        print(f'dataset={name}, split={split}, corrcoef={corrcoef}')


def build_model(base_dir, min_window_size=16, window_size=32, strategy='cell+first-last-avg'):
    _, rannet_model = RanNet.load_rannet(
        config_path=f'{base_dir}/config.json',
        checkpoint_path=f'{base_dir}/model.ckpt',
        return_sequences=True,
        return_cell=True,
        apply_cell_transform=False,
        window_size=window_size,
        min_window_size=min_window_size,
        cell_pooling='mean'
    )

    if strategy == 'cell':
        output = rannet_model.output[1]
    elif strategy == 'first-last-cell-avg':
        output = L.Average()([
            rannet_model.get_layer('RAN-0').output[1],
            rannet_model.get_layer('RAN-1').output[1]
        ])
    elif strategy == 'cell+first-last-maxpool-avg':
        first_output = L.Lambda(lambda x: x[0] + x[1])([
            L.GlobalMaxPooling1D()(rannet_model.get_layer('RAN-0').output[0]),
            rannet_model.get_layer('RAN-0').output[1],
        ])
        final_output = L.Lambda(lambda x: x[0] + x[1])([
            L.GlobalMaxPooling1D()(rannet_model.get_layer('RAN-1').output[0]),
            rannet_model.get_layer('RAN-1').output[1],
        ])
        output = L.Average()([first_output, final_output])
    elif strategy == 'cell+first-last-avgpool-avg':
        first_output = L.Average()([
            rannet_model.get_layer('RAN-0').output[1],
            rannet_model.get_layer('RAN-1').output[1],
        ])
        final_output = L.Average()([
            L.GlobalAveragePooling1D()(rannet_model.get_layer('RAN-0').output[0]),
            L.GlobalAveragePooling1D()(rannet_model.get_layer('RAN-1').output[0]),            
        ])
        output = L.Lambda(lambda x: x[0] + x[1])([first_output, final_output])
    elif strategy == 'first-last-maxpool-avg':
        output = L.Average()([
            L.GlobalMaxPooling1D()(rannet_model.get_layer('RAN-0').output[0]),
            L.GlobalMaxPooling1D()(rannet_model.get_layer('RAN-1').output[0])
        ])
    elif strategy == 'first-last-avgpool-avg':
        output = L.Average()([
            L.GlobalAveragePooling1D()(rannet_model.get_layer('RAN-0').output[0]),
            L.GlobalAveragePooling1D()(rannet_model.get_layer('RAN-1').output[0])
        ])
    model = keras.Model(rannet_model.input, [output])
    model.summary()
    return model


if __name__ == '__main__':
    base_dir = '../../../RAN-Pretrained-Models/rannet-base-v2-cn-uncased-general-model'
    '''
    LCQMC = load_dataset("shibing624/nli_zh", "LCQMC")
    evaluate(base_dir, LCQMC, 'LCQMC', min_window_size=10, window_size=20, strategy='cell+first-last-avgpool-avg', apply_whitening=False, n_components=512)
    '''
    BQ = load_dataset("shibing624/nli_zh", "BQ")
    evaluate(base_dir, BQ, 'BQ', min_window_size=10, window_size=32,
             strategy='first-last-avgpool-avg', apply_whitening=False,
             n_components=32)  # best: first-last-avgpool-avg
    '''
    ATEC = load_dataset("shibing624/nli_zh", "ATEC")
    evaluate(base_dir, ATEC, 'ATEC', min_window_size=10, window_size=20, strategy='cell+first-last-avgpool-avg', apply_whitening=False, n_components=128)
    '''
