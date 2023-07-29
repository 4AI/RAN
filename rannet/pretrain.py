# -*- coding: utf-8 -*-


""" Pretrain RanNet
"""

import os
import re
import json
import time
import random

import click
import tensorflow as tf
from langml import keras, TF_KERAS, log
from tqdm import tqdm
from boltons.iterutils import chunked_iter

from rannet import RanNetForMLMPretrain, RanNetParams
from rannet.dataloader import DataLoader, BertMlmDataLoader
from rannet.tokenizer import RanNetWordPieceTokenizer


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def split_sentences(text: str, max_length: int = 512, lang: str = 'english'):
    def get_length(t):
        return len(t.split()) if lang == 'english' else len(t)

    texts = re.findall(r'.+?[。？！\n\.!?]', text)
    cache_text = ''
    for text in texts:
        if get_length(cache_text + text) > max_length:
            if cache_text:
                yield cache_text
            cache_text = text
        else:
            cache_text += text
    if cache_text:
        yield cache_text


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """rannet client"""
    pass


@cli.command()
@click.option('--vocab_path', type=str, required=True, help='Specify vocabulary path.')
@click.option('--workers', type=int, default=2, help='Specify workers. Defaults to 2')
@click.option('--min_length', type=int, default=1, help='Specify min sequence length. Defaults to 1')
@click.option('--max_length', type=int, default=3000, help='Specify max sequence length. Defaults to 3000')
@click.option('--chunk_size', type=int, default=10000, help='Specify data chunk size. Defaults to 10000')
@click.option('--corpus_dir', type=str, required=True, help='Specify corpus dir.')
@click.option('--save_dir', type=str, required=True, help='Specify save dir.')
@click.option('--whole_word_tokenizer', type=str, default='whitespace',
              help='Specify whole_word_tokenizer from [`whitespace`, `jieba`]. Defaults to `whitespace`')
@click.option('--cased', is_flag=True, default=False, help='use --cased to set lowercase=False.')
def corpus(vocab_path: str, workers: int, min_length: int, max_length: int, chunk_size: int,
           corpus_dir: str, save_dir: str, whole_word_tokenizer: str, cased: bool):
    # make dirs
    os.makedirs(save_dir, exist_ok=True)

    # init tokenizer
    lowercase = False if cased else True
    log.info(f'lowercase={lowercase}')
    tokenizer = RanNetWordPieceTokenizer(vocab_path, lowercase=lowercase)
    tokenizer.enable_truncation(max_length=max_length)

    # init dataloder
    assert whole_word_tokenizer in ['jieba', 'whitespace']
    lang = 'english'
    if whole_word_tokenizer == 'jieba':
        import jieba_fast as jieba  # NOQA
        word_segment = lambda x: jieba.lcut(x, HMM=False)  # NOQA
        lang = 'chinese'
    else:
        word_segment = lambda x: x.split()  # NOQA
    dataloader = BertMlmDataLoader(tokenizer, word_segment, max_length=max_length)

    log.info('start to load data...')
    data_size = 0
    for fname in os.listdir(corpus_dir):
        corpus_path = os.path.join(corpus_dir, fname)
        save_path = os.path.join(save_dir, '.'.join(fname.split('.')[:-1]) + '.tfrecords')
        log.info(f'process {corpus_path}')
        data = []
        with open(corpus_path, 'r') as reader:
            for line in tqdm(reader):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj['text']
                if not isinstance(text, str):
                    continue
                for sentence in split_sentences(text, max_length=max_length, lang=lang):
                    sentence = sentence.strip()
                    if whole_word_tokenizer == 'whitespace':
                        if len(sentence.split()) < min_length:
                            continue
                    else:
                        if len(sentence) < min_length:
                            continue
                    data.append({'sentence': sentence})
        log.info(f'data size: {len(data)}')
        data_size += len(data)
        data = chunked_iter(data, chunk_size)
        dataloader.process(data, save_path, workers=workers)
        log.info(f'save to {save_path}')
    info_path = os.path.join(save_dir, 'record_info.json')
    log.info(f'final data size {data_size}, info has saved to {info_path}')
    with open(info_path, 'w') as writer:
        writer.writelines(json.dumps({'data_size': data_size}, ensure_ascii=False) + '\n')


@cli.command()
@click.option('--vocab_path', type=str, required=True, help='Specify vocabulary path.')
@click.option('--max_length', type=int, default=3000, help='Specify max sequence length. Defaults to 3000')
@click.option('--chunk_size', type=int, default=10000, help='Specify data chunk size. Defaults to 10000')
@click.option('--corpus_path', type=str, required=True, help='Specify corpus path.')
@click.option('--save_path', type=str, required=True, help='Specify save path.')
def single_corpus(vocab_path: str, max_length: int, chunk_size: int, corpus_path: str, save_path: str):
    tokenizer = RanNetWordPieceTokenizer(vocab_path)
    tokenizer.enable_truncation(max_length=max_length)
    dataloader = BertMlmDataLoader(tokenizer, lambda x: x.split())

    log.info('start to load data...')
    data = []
    with open(corpus_path, 'r') as reader:
        for line in tqdm(reader):
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    log.info('convert to tfrecord...')
    data = chunked_iter(data, chunk_size)
    dataloader.process(data, save_path)


@cli.command()
@click.option('--corpus_path', type=str, required=True, help='Specify tfrecord corpus dir/path.')
@click.option('--config_path', type=str, required=True, help='Specify hyperparameter config path.')
@click.option('--log_path', type=str, default=None,
              help='Specify traning log path. Defaults to trainig.{timestamp}.log')
@click.option('--base_ckpt_path', type=str, default=None,
              help='Specify the checkpoint path of the base RanNet model when continuing pretraining.')
@click.option('--save_dir', type=str, required=True, help='Specify dir to save model')
@click.option('--record_info_path', type=str, required=True, help='Specify record info path.')
@click.option('--batch_size', type=int, default=32, help='Specify batch size. Defaults to 32')
@click.option('--learning_rate', type=float, default=1e-3, help='Specify learning rate. Defaults to 1e-3')
@click.option('--weight_decay', type=float, default=0.01, help='Specify weight decay. Defaults to 1e-2')
@click.option('--sequence_length', type=int, default=512, help='Specify sequence length. Defaults to 512')
@click.option('--num_warmup_steps', type=int, default=3000, help='Specify warmup steps. Defaults to 3000')
@click.option('--num_train_steps', type=int, default=125000, help='Specify training steps. Defaults to 125000')
@click.option('--ckpt_save_freq', type=int, default=125000, help='Specify ckpt save freq. Defaults to 125000')
@click.option('--gradient_accumulation_steps', type=int, default=1,
              help='Specify gradient accumulation steps. Defaults to 1, if value > 1, then use gradient accumulation.')
@click.option('--distributed', is_flag=True, default=False, help='Specify distributed training with `--distributed`.')
@click.option('--distributed_strategy', type=str, default='MirroredStrategy',
              help='Specify distributed training strategy. Defaults to MirroredStrategy')
@click.option('--verbose', type=int, default=1,
              help='0 = silent, 1 = progress bar, 2 = one line per epoch.')
def pretrain(corpus_path: str, config_path: str, log_path: str, base_ckpt_path: str, save_dir: str,
             record_info_path: str, batch_size: int, learning_rate: float, weight_decay: float,
             sequence_length: int, num_warmup_steps: int, num_train_steps: int, ckpt_save_freq: int,
             gradient_accumulation_steps: int, distributed: bool, distributed_strategy: str, verbose: int):
    # create save dir
    os.makedirs(save_dir, exist_ok=True)

    # environment check
    assert TF_KERAS, 'To pretrain rannet, please `export TF_KERAS=1` first'

    # load hyperparameters
    with open(config_path, 'r') as reader:
        config = json.load(reader)
    params = RanNetParams(config)
    log.info(f'config: {config}')

    # load model
    def build_model():
        rannet = RanNetForMLMPretrain(params)
        model = rannet()

        lr_schedule = None if gradient_accumulation_steps < 2 else {
            num_warmup_steps * gradient_accumulation_steps: 1.0,
            num_train_steps * gradient_accumulation_steps: 0.0,
        }
        grad_accum_steps = None if gradient_accumulation_steps < 2 else gradient_accumulation_steps
        RanNetForMLMPretrain.compile(model,
                                     learning_rate=learning_rate,
                                     weight_decay=weight_decay,
                                     loss=None,
                                     lr_schedule=lr_schedule,
                                     gradient_accumulation_steps=grad_accum_steps,
                                     metrics=['sparse_categorical_accuracy'])

        if base_ckpt_path is not None:
            # restore the pre-trained model
            log.info(f'restore weights from {base_ckpt_path}')
            try:
                model.load_weights(base_ckpt_path)
            except NotImplementedError:
                log.warn('failed to restore weights via `model.load_weights`, '
                         'change to `rannet.restore_weights_from_checkpoint`')
                rannet.restore_weights_from_checkpoint(model, base_ckpt_path)
                log.info('successfully restore weights!')
        return model

    if distributed:
        strategy = getattr(tf.distribute, distributed_strategy)()
        with strategy.scope():
            train_model = build_model()
    else:
        train_model = build_model()

    with open(record_info_path, 'r') as reader:
        record_info = json.load(reader)

    data_size = record_info['data_size']
    # training setting
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch < data_size / batch_size:
        steps_per_epoch += 1
    epochs = num_train_steps // steps_per_epoch
    log_path = f'traning.{int(time.time())}.log' if log_path is None else log_path
    log_callback = keras.callbacks.CSVLogger(log_path)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, '{epoch:02d}.weights'),
        save_weights_only=True,
        save_freq=ckpt_save_freq,
    )

    if os.path.isdir(corpus_path):
        record_paths = []
        for fname in os.listdir(corpus_path):
            if fname.endswith('tfrecords'):
                record_paths.append(os.path.join(corpus_path, fname))
    else:
        record_paths = [corpus_path]
    random.shuffle(record_paths)
    log.info(f'record size: {len(record_paths)}')

    # load dataset
    dataset = DataLoader.load_tfrecord(
        record_paths, batch_size // gradient_accumulation_steps, sequence_length=sequence_length)

    # start to train
    train_model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=[checkpoint_callback, log_callback],
    )


@cli.command()
@click.option('--config_path', type=str, required=True, help='Specify config path.')
@click.option('--ckpt_path', type=str, required=True, help='Specify checkpoint path.')
@click.option('--target_path', type=str, required=True, help='Specify target checkpoint path.')
def export_checkpoint(config_path: str, ckpt_path: str, target_path: str):
    from rannet import RanNet

    RanNet.export_checkpoint(config_path, ckpt_path, target_path)
    log.info(f'successfully exported to {target_path}')


def main():
    cli(prog_name='rannet-cli', obj={})


if __name__ == '__main__':
    main()
