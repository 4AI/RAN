# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from scipy.stats import spearmanr
from langml.utils import pad_sequences
from rannet import RanNet, RanNetWordPieceTokenizer


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


def evaluate(tokenizer, model, dataset, name, apply_whitening=True, n_components=768):
    for split in ['test']: # ['train', 'validation', 'test']:
        print(f'evaluate {name}-{split}')
        a_token_ids, b_token_ids, labels = convert2ids(tokenizer, dataset[split])
        a_token_ids = a_token_ids[:100, :]
        b_token_ids = b_token_ids[:100, :]
        labels = labels[:100]
        print('a_token_ids shape:', a_token_ids.shape)
        print('b_token_ids shape:', b_token_ids.shape)
        a_vecs = model.predict(a_token_ids, verbose=True)
        b_vecs = model.predict(b_token_ids, verbose=True)
        print('a_vecs shape:', a_vecs.shape)
        print('b_vecs shape:', b_vecs.shape)
        # whitening
        if apply_whitening:
            all_vecs = [(a_vecs, b_vecs)]
            kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])
            kernel = kernel[:, :n_components]
            a_vecs = transform_and_normalize(a_vecs, kernel, bias)
            b_vecs = transform_and_normalize(b_vecs, kernel, bias)
        else:
            a_vecs = l2_normalize(a_vecs)
            b_vecs = l2_normalize(b_vecs)

        sims = (a_vecs * b_vecs).sum(axis=1)
        corrcoef = spearmanr(labels, sims).correlation

        print(f'dataset={name}, split={split}, corrcoef={corrcoef}')


if __name__ == '__main__':
    base_dir = '../../../RAN-Pretrained-Models/rannet-small-v2-cn-uncased-general-model'
    tokenizer = RanNetWordPieceTokenizer(f'{base_dir}/vocab.txt', lowercase=True)
    _, rannet_model = RanNet.load_rannet(
        config_path=f'{base_dir}/config.json',
        checkpoint_path=f'{base_dir}/model.ckpt',
        return_sequences=False,
        apply_cell_transform=False,
    )
    rannet_model.summary()
    ATEC = load_dataset("shibing624/nli_zh", "ATEC")
    evaluate(tokenizer, rannet_model, ATEC, 'ATEC')
    # BQ = load_dataset("shibing624/nli_zh", "BQ")
    # evaluate(tokenizer, rannet_model, BQ, 'BQ')
