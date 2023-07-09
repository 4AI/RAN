# -*- coding: utf-8 -*-

import os
import shutil
import zipfile

import numpy as np
import gdown


RANNET_NAME = 'rannet-small-v2-cn-uncased-general-model.zip'
RANNET_DIR = 'rannet-small-v2-cn-uncased-general-model'


def setup_module(module):
    if not os.path.exists(RANNET_DIR):
        gdown.download(id='1D-FCxY_UMwZCkvcwl6hkRcl6VnCzRGIj', output=RANNET_NAME, quiet=False)
        with zipfile.ZipFile(RANNET_NAME, 'r') as zip_ref:
            zip_ref.extractall('./')


def teardown_module(module):
    shutil.rmtree(RANNET_DIR)
    os.remove(RANNET_NAME)


def test_load_rannet():
    from rannet import RanNet, RanNetWordPieceTokenizer

    vocab_path = os.path.join(RANNET_DIR, 'vocab.txt')
    ckpt_path = os.path.join(RANNET_DIR, 'model.ckpt')
    config_path = os.path.join(RANNET_DIR, 'config.json')
    tokenizer = RanNetWordPieceTokenizer(vocab_path, lowercase=True)

    rannet, rannet_model = RanNet.load_rannet(
        config_path=config_path,
        checkpoint_path=ckpt_path,
        return_sequences=False,
        return_cell=True,
        apply_cell_transform=False,
        cell_pooling='mean')

    vec = rannet_model.predict(np.array([tokenizer.encode('rannet').ids]))
    assert vec.shape == (1, 128)
