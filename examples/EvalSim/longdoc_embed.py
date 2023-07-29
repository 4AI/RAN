# -*- coding: utf-8 -*-

import numpy as np
from rannet import RanNet, RanNetWordPieceTokenizer
from scipy import spatial


base_dir = '/Users/seanlee/Workspace/RAN-Pretrained-Models/rannet-base-v3-en-uncased-model'
vocab_path = f'{base_dir}/vocab.txt'
ckpt_path = f'{base_dir}/model.ckpt'
config_path = f'{base_dir}/config.json'
embedding_size = 768
tokenizer = RanNetWordPieceTokenizer(vocab_path, lowercase=True)

rannet, rannet_model = RanNet.load_rannet(
    config_path=config_path,
    checkpoint_path=ckpt_path,
    return_sequences=False,
    return_history=True,
    apply_cell_transform=False,
    cell_pooling='mean',
    with_cell=True,
    window_size=32,
)

text = '''William Shakespeare's name is synonymous with many of the famous lines he wrote in his plays and prose. Yet his poems are not nearly as recognizable to many as the characters and famous monologues from his many plays.'''

tok = tokenizer.encode(text)
x = np.array([tok.ids])
vec, history = rannet_model.predict([x, np.zeros((1, embedding_size))])

text = 'hello world'
tok = tokenizer.encode(text)
x2 = np.array([tok.ids])
vec2, _ = rannet_model.predict([x2, np.zeros((1, embedding_size))])
vec3, _ = rannet_model.predict([x2, history])
print(vec2 == vec3)

print('cos sim (v1, v2):', 1 - spatial.distance.cosine(vec.reshape(-1), vec2.reshape(-1)))
print('cos sim (v1, v3):', 1 - spatial.distance.cosine(vec.reshape(-1), vec3.reshape(-1)))
print('cos sim (v2, v3):', 1 - spatial.distance.cosine(vec2.reshape(-1), vec3.reshape(-1)))
