# -*- coding: utf-8 -*-

""" DataLoader for pretraininng
"""
import concurrent.futures
from typing import Dict, Optional, Callable, List, Tuple, Union

import numpy as np
import tensorflow as tf
from langml import K
from rannet.tokenizer import RanNetWordPieceTokenizer


def subfinder(array: List, sub_array: List) -> List[int]:
    """ find sub-array positions
    example:
    >>> array = [0, 0, 1, 2, 3, 5, 1, 2, 3, 1, 2]
    >>> sub_array = [1, 2, 3]
    >>> subfinder(array, sub_array)
    [2, 6]
    """
    indices = []
    for i in range(len(array)):
        if array[i] == sub_array[0] and array[i:i + len(sub_array)] == sub_array:
            indices.append(i)
    return indices


class DataLoader:
    """ dataloader for pretraning
    """
    def __init__(
        self,
        tokenizer: RanNetWordPieceTokenizer,
        max_length: int = 512
    ):
        """
        Args:
            tokenizer: RanNetWordPieceTokenizer
            max_length: Optional[int], specify max sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_mask_id = tokenizer.token_to_id(tokenizer.special_tokens.MASK)
        self.vocab_size = tokenizer.get_vocab_size()

    def process_sentence(self, text) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

    def tfrecord_serialize(self, instances, instance_keys=['token_ids', 'mask_ids']):
        """ convert to tfrecord
        """
        def create_feature(x):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

        serialized_instances = []
        for instance in instances:
            features = {
                k: create_feature(v)
                for k, v in zip(instance_keys, instance)
            }
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            serialized_instance = tf_example.SerializeToString()
            serialized_instances.append(serialized_instance)

        return serialized_instances

    @staticmethod
    def load_tfrecord(record_paths, batch_size, sequence_length=512, buffer_size=None):
        """ load dataset from tfrecord
        """
        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature(shape=[sequence_length], dtype=tf.int64),
                'mask_ids': tf.io.FixedLenFeature(shape=[sequence_length], dtype=tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
            mlm_mask = K.not_equal(mask_ids, 0)
            masked_token_ids = K.cast(K.switch(mlm_mask, mask_ids, token_ids), K.floatx())
            token_ids = K.cast(token_ids, K.floatx())
            x = {
                'Input-Token': masked_token_ids,
                'token_ids': token_ids,
                'mlm_mask': K.cast(mlm_mask, K.floatx()),
            }

            return x, token_ids

        if not isinstance(record_paths, list):
            record_paths = [record_paths]

        dataset = tf.data.TFRecordDataset(record_paths)
        dataset = dataset.map(parse_function)
        dataset = dataset.repeat()
        if buffer_size is None:
            buffer_size = batch_size * 1000
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    def get_random_token(self, token_id: int) -> int:
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        while True:
            token_id = np.random.randint(0, self.vocab_size)
            token = self.tokenizer.id_to_token(token_id)
            if token in self.tokenizer.special_tokens:
                continue
            return token_id

    def truncate_pad_sequence(self, sequence: List[int], padding_value=0) -> List[int]:
        sequence = sequence + [padding_value] * (self.max_length - len(sequence))
        return sequence[:self.max_length]

    def process_paragraph(self, texts: Union[List[str], List[Dict[str, str]]]):
        """
        Args:
            texts: Union[List[str], List[Dict[str, str]]]
                for NOLAN-Style: [{"word": "xxx", "sentence": "xxx"}, ],
                for BERT-Style: ["sentence 1", "xxx", ]
        """
        instances = []
        for text in texts:
            instance = self.process_sentence(text)
            if instance is None:
                continue
            instances.append(instance)

        return instances

    def process(self, corpus: List[List], record_path: str, workers=4):
        """ process corpus
        """
        def write_to_tfrecord(serialized_instances):
            for serialized_instance in serialized_instances:
                writer.write(serialized_instance)

        def process_paragraph(texts):
            instances = self.process_paragraph(texts)
            serialized_instances = self.tfrecord_serialize(instances) if instances else None
            return serialized_instances

        with tf.io.TFRecordWriter(record_path) as writer:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                for serialized_instances in executor.map(process_paragraph, corpus):
                    if serialized_instances is not None:
                        write_to_tfrecord(serialized_instances)

        print(f'Records have saved to {record_path}')


class BertMlmDataLoader(DataLoader):
    """ DataLoader with BERT MLM setting
    """
    def __init__(
        self,
        tokenizer: RanNetWordPieceTokenizer,
        word_segment: Callable,
        mask_rate: float = 0.15,
        max_length: int = 512
    ):
        """
        Args:
            tokenizer: RanNetWordPieceTokenizer,
            word_segment: Optional[Callable]. word segmentation function to support the whole word mask strategy
            mask_rate: float,
            max_length: Optional[int],
        """
        super(BertMlmDataLoader, self).__init__(
            tokenizer, max_length=max_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate

    def process_sentence(self, obj: str) -> Tuple[List[int], List[int]]:
        text = obj['sentence']
        words = self.word_segment(text)
        probas = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for word, proba in zip(words, probas):
            encoded = self.tokenizer.encode(word)
            word_tokens = encoded.tokens
            word_token_ids = encoded.ids
            token_ids.extend(word_token_ids)
            mask_ids += ([self.get_random_token(i) for i in word_token_ids]
                         if proba < self.mask_rate else [0] * len(word_tokens))

        token_ids = self.truncate_pad_sequence(token_ids)
        mask_ids = self.truncate_pad_sequence(mask_ids)
        return token_ids, mask_ids


class Seq2SeqLMDataLoader:
    """ DataLoader for seq2seq
    """
    def __init__(
        self,
        tokenizer: RanNetWordPieceTokenizer,
        mask_rate: float = 0.15,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None
    ):
        """
        Args:
            tokenizer: RanNetWordPieceTokenizer,
            mask_rate: float,
            max_length: Optional[int],
        """
        self.tokenizer = tokenizer
        if max_source_length is not None:
            self.tokenizer.enable_truncation(max_length=max_source_length)
        self.mask_rate = mask_rate
        self.max_target_length = max_target_length

        self.token_mask_id = self.tokenizer.token_to_id(self.tokenizer.special_tokens.MASK)
        self.token_sep_id = self.tokenizer.token_to_id(self.tokenizer.special_tokens.SEP)

    def process(self, source_text: str, target_text: str) -> Tuple[List[int], List[int], List[int]]:
        source_tok = self.tokenizer.encode(source_text)
        source_ids = source_tok.ids

        words = list(target_text)
        probas = np.random.random(len(words))
        target_ids, target_mask_ids = [], []
        for word, proba in zip(words, probas):
            encoded = self.tokenizer.encode(word)
            word_token_ids = encoded.ids
            target_ids.extend(word_token_ids)
            target_mask_ids += ([self.token_mask_id] * len(word_token_ids)
                                if proba < self.mask_rate else word_token_ids)
        if self.max_target_length is not None:
            target_ids = target_ids[:self.max_target_length - 1]
            target_mask_ids = target_mask_ids[:self.max_target_length - 1]
        token_ids = source_ids + [self.token_sep_id] + target_ids + [self.token_sep_id]
        segment_ids = [0] * (len(source_ids) + 1) + [1] * (len(target_ids) + 1)
        # target_mask_ids[0] = self.token_mask_id
        mask_ids = source_ids + [self.token_sep_id] + target_mask_ids + [self.token_mask_id]
        return token_ids, segment_ids, mask_ids
