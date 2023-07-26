# -*- coding: utf-8 -*-

""" lastest version
"""

import json
import itertools
from typing import Dict, Union, Optional, Callable, Tuple, List, Any

import tensorflow as tf
from langml import keras, L, K, TF_KERAS
from langml.layers import LayerNorm
from langml.plm import TokenEmbedding, EmbeddingMatching
from langml.tensor_typing import Models, Tensors
from langml.utils import load_variables
from langml import log

from rannet.layers import (
    AdaptiveEmbedding, AdaptiveSoftmax,
    WithSparseCategoricalCrossEntropy,
    WithSparseCategoricalAccuracy,
)
from rannet.ran import RAN, GatedLinearUnit
from rannet.optimizer import AdamWarmup


class RanNetParams:
    def __init__(self, config: Dict):
        self.vocab_size = config['vocab_size']
        self.embedding_size = config.get('embedding_size', 512)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.ran_layers = config.get('ran_layers', 2)
        self.head_size = config.get('head_size', 128)
        self.head_num = config.get('head_num', 8)
        self.window_size = config.get('window_size', 256)
        self.min_window_size = config.get('min_window_size', 16)
        self.cell_pooling = config.get('cell_pooling', 'last')  # last | mean | max
        self.embedding_initializer = config.get('embedding_initializer', 'truncated_normal')
        self.kernel_initializer = config.get('kernel_initializer', 'truncated_normal')
        self.kernel_initializer_range = config.get('kernel_initializer_range', 0.02)

    @staticmethod
    def from_file(config_path: str):
        with open(config_path, 'r') as reader:
            config = json.load(reader)
        return RanNetParams(config)

    def __str__(self):
        return ','.join([f'{k}={v}' for k, v in self.__dict__.items()])


class RanNet:
    def __init__(self,
                 params: RanNetParams,
                 return_sequences: bool = True,
                 return_cell: bool = True,
                 return_history: bool = False,
                 mlm_softmax: bool = False,
                 apply_cell_transform: bool = True,
                 apply_lm_mask: bool = False,
                 apply_seq2seq_mask: bool = False,
                 apply_memory_review: bool = True,
                 cell_pooling: str = 'last',
                 min_window_size: Optional[int] = None,
                 window_size: Optional[int] = None,
                 prefix: str = ''):
        self.params = params
        self.return_sequences = return_sequences
        self.return_cell = return_cell
        self.return_history = return_history
        self.apply_cell_transform = apply_cell_transform
        self.cell_pooling = cell_pooling
        self.min_window_size = min_window_size
        self.window_size = window_size
        self.prefix = prefix
        self.inputs = None
        self.__var_status = {}

        if self.params.kernel_initializer == 'truncated_normal':
            self.initializer = keras.initializers.TruncatedNormal(
                stddev=self.params.kernel_initializer_range)
        else:
            self.initializer = self.params.kernel_initializer
        if self.params.embedding_initializer == 'truncated_normal':
            self.embedding_initializer = keras.initializers.TruncatedNormal(
                stddev=self.params.kernel_initializer_range)
        else:
            self.embedding_initializer = self.params.embedding_initializer
        self.text_masking = L.Lambda(
            lambda t: K.cast(K.greater(t, 0), K.floatx()), name=self.get_weight_name('Text-Mask'))
        self.token_embedding = TokenEmbedding(
            input_dim=self.params.vocab_size,
            output_dim=self.params.embedding_size,
            mask_zero=True,
            embeddings_initializer=self.embedding_initializer,
            name=self.get_weight_name('Token-Embedding')
        )
        self.embedding_layernorm = LayerNorm(name=self.get_weight_name('Embedding-LayerNorm'))
        self.embedding_dropout = L.Dropout(self.params.dropout_rate, name=self.get_weight_name('Embedding-Dropout'))
        self.rans = [
            RAN(
                self.params.head_num,
                head_size=self.params.head_size,
                window_size=self.window_size or self.params.window_size,
                min_window_size=self.min_window_size or self.params.min_window_size,
                activation='swish',
                kernel_initializer=self.initializer,
                apply_lm_mask=apply_lm_mask,
                apply_seq2seq_mask=apply_seq2seq_mask,
                apply_memory_review=apply_memory_review,
                dropout_rate=self.params.dropout_rate,
                cell_pooling=self.cell_pooling or self.params.cell_pooling,
                name=self.get_weight_name(f'RAN-{i}')
            )
            for i in range(self.params.ran_layers)
        ]
        self.mlm_hidden = keras.Sequential([
            L.Dropout(self.params.dropout_rate, name='Dropout'),
            L.Dense(
                self.params.embedding_size,
                activation='swish',
                kernel_initializer=self.initializer,
                name='Dense'
            ),
            LayerNorm(name='LayerNorm'),
        ], name=self.get_weight_name('MLM-Hidden'))
        self.mlm_matching = EmbeddingMatching(use_softmax=mlm_softmax, name=self.get_weight_name('MLM-Matching'))

    def get_weight_name(self, name: str) -> str:
        prefix = self.prefix.rstrip('-')
        if prefix:
            return f'{prefix}-{name}'
        return name

    def remove_prefix(self, name: str) -> str:
        if self.prefix:
            prefix = self.prefix.rstrip('-')
            return name.replace(f'{prefix}-', '')
        return name

    def get_inputs(self, with_cell: bool = False, with_segment: bool = False):
        x_in = L.Input(name=self.get_weight_name('Input-Token'), shape=(None, ))
        x = x_in
        self.inputs = [x_in]
        if with_cell:
            cell_in = L.Input(name=self.get_weight_name('Input-Cell'), shape=(self.params.embedding_size, ))
            self.inputs.append(cell_in)
            cell = cell_in
        else:
            cell = None
        if with_segment:
            segment_in = L.Input(name=self.get_weight_name('Input-Segment'), shape=(None, ))
            self.inputs.append(segment_in)
            segments = segment_in
        else:
            segments = None
        return x, cell, segments

    def encode(self,
               x: Tensors,
               x_mask: Tensors,
               cell: Optional[Tensors] = None,
               segments: Optional[Tensors] = None) -> Union[Tensors, List[Tensors]]:
        x = self.embedding_layernorm(x)
        x = self.embedding_dropout(x)

        outputs = x
        last_cell = cell
        for kernel in self.rans:
            last_cell, outputs, cell = kernel([outputs, x_mask], cell=last_cell, segments=segments)

        if self.return_cell and self.apply_cell_transform:
            cell = L.Lambda(lambda x: K.expand_dims(x, axis=1))(cell)
            cell = L.Dense(self.params.embedding_size,
                           kernel_initializer=self.initializer,
                           activation='swish',
                           name='Output-Cell-Dense')(cell)
            cell = L.Dropout(self.params.dropout_rate)(cell)
            cell = GatedLinearUnit(
                self.params.embedding_size,
                kernel_initializer=self.initializer,
                name='Output-Cell-GLU')(cell)
            max_pooling = L.Lambda(lambda x: K.max(x, axis=1, keepdims=True))(outputs)
            max_pooling = L.Dense(self.params.embedding_size,
                                  kernel_initializer=self.initializer,
                                  name='Output-Pooling-Dense')(max_pooling)
            # ct = p + g(c), use maxpooling to enhance semantic feature
            cell = L.Lambda(lambda x: x[0] + x[1], name='Output-Cell-Fuse')([cell, max_pooling])
            cell = L.Lambda(lambda x: K.squeeze(x, axis=1), name='Output-Cell-Squeeze')(cell)

        ret = []
        if self.return_sequences:
            ret.append(outputs)
        if self.return_cell:
            ret.append(cell)
        if self.return_history:
            ret.append(last_cell)
        return ret

    def __call__(self,
                 with_cell: bool = False,
                 with_mlm: bool = False,
                 return_model: bool = True,
                 seq2seq: bool = False) -> Union[Models, Tensors]:
        """ build model
        Args:
            with_cell: bool. Whether to input with cell. Defaults to False
            with_mlm: bool,  if set `True` the mlm outputs will be returned,
                         if set `False`, the finetuning outputs including the sentence representation
                         and the semantic cell will be returned. Defaults to False.
            return_model: bool, whether to return keras model. Defaults to True.
            seq2seq: bool. Set seq2seq model. if seq2seq is True, segment information is required to input.
                Segment is used to compute prefix causal mask. Defaults to False.
        """
        self.__var_status['with_cell'] = with_cell
        self.__var_status['with_mlm'] = with_mlm
        self.__var_status['return_model'] = return_model
        self.__var_status['seq2seq'] = seq2seq

        if seq2seq:
            assert with_mlm, "seq2seq only works for mlm model, please specify `with_mlm=True`"
        if with_mlm:
            assert self.return_sequences, "please specify `RanNet(..., return_sequences=True)`, when return mlm"
        x, cell, segments = self.get_inputs(with_cell=with_cell, with_segment=seq2seq)
        x_mask = self.text_masking(x)

        x, embedding_weights = self.token_embedding(x)
        outputs = self.encode(x, x_mask, cell, segments=segments)

        if not with_mlm:
            if return_model:
                return keras.Model(self.inputs, outputs)
            return outputs

        output = outputs[0]
        mlm = self.mlm_hidden(output)
        mlm = self.mlm_matching([mlm, embedding_weights])

        if return_model:
            return keras.Model(self.inputs, mlm)
        return mlm

    @staticmethod
    def variable_mapping(prefix: str = 'rannet', ran_layers: int = 2) -> Dict:
        mapping = {
            'Token-Embedding': [f'{prefix}/token-embedding/embeddings'],
            'Embedding-LayerNorm': [
                f'{prefix}/embedding-layernorm/beta',
                f'{prefix}/embedding-layernorm/gamma',
            ],
        }
        for i in range(ran_layers):
            mapping[f'RAN-{i}'] = [
                f'{prefix}/ran-{i}/cell-glu/dense-t/bias',
                f'{prefix}/ran-{i}/cell-glu/dense-t/kernel',
                f'{prefix}/ran-{i}/cell-glu/dense-g/bias',
                f'{prefix}/ran-{i}/cell-glu/dense-g/kernel',

                f'{prefix}/ran-{i}/cell-initializer/kernel',
                f'{prefix}/ran-{i}/cell-initializer-layernorm/beta',
                f'{prefix}/ran-{i}/cell-initializer-layernorm/gamma',

                f'{prefix}/ran-{i}/cell-residual-layernorm/beta',
                f'{prefix}/ran-{i}/cell-residual-layernorm/gamma',

                f'{prefix}/ran-{i}/concat-layernorm/beta',
                f'{prefix}/ran-{i}/concat-layernorm/gamma',

                f'{prefix}/ran-{i}/encode-attn/glu/dense-t/bias',
                f'{prefix}/ran-{i}/encode-attn/glu/dense-t/kernel',
                f'{prefix}/ran-{i}/encode-attn/glu/dense-g/bias',
                f'{prefix}/ran-{i}/encode-attn/glu/dense-g/kernel',
                f'{prefix}/ran-{i}/encode-attn/dense-k/bias',
                f'{prefix}/ran-{i}/encode-attn/dense-k/kernel',
                f'{prefix}/ran-{i}/encode-attn/layernorm/beta',
                f'{prefix}/ran-{i}/encode-attn/layernorm/gamma',
                f'{prefix}/ran-{i}/encode-attn/dense-o/bias',
                f'{prefix}/ran-{i}/encode-attn/dense-o/kernel',
                f'{prefix}/ran-{i}/encode-attn/dense-q/bias',
                f'{prefix}/ran-{i}/encode-attn/dense-q/kernel',
                f'{prefix}/ran-{i}/encode-attn/dense-v/bias',
                f'{prefix}/ran-{i}/encode-attn/dense-v/kernel',

                f'{prefix}/ran-{i}/output-attn/glu/dense-t/bias',
                f'{prefix}/ran-{i}/output-attn/glu/dense-t/kernel',
                f'{prefix}/ran-{i}/output-attn/glu/dense-g/bias',
                f'{prefix}/ran-{i}/output-attn/glu/dense-g/kernel',
                f'{prefix}/ran-{i}/output-attn/dense-k/bias',
                f'{prefix}/ran-{i}/output-attn/dense-k/kernel',
                f'{prefix}/ran-{i}/output-attn/layernorm/beta',
                f'{prefix}/ran-{i}/output-attn/layernorm/gamma',
                f'{prefix}/ran-{i}/output-attn/dense-o/bias',
                f'{prefix}/ran-{i}/output-attn/dense-o/kernel',
                f'{prefix}/ran-{i}/output-attn/dense-q/bias',
                f'{prefix}/ran-{i}/output-attn/dense-q/kernel',
                f'{prefix}/ran-{i}/output-attn/dense-v/bias',
                f'{prefix}/ran-{i}/output-attn/dense-v/kernel',
            ]
        mapping = dict(mapping, **{
            'MLM-Hidden': [
                f'{prefix}/mlm-hidden/dense/bias',
                f'{prefix}/mlm-hidden/dense/kernel',
                f'{prefix}/mlm-hidden/layernorm/beta',
                f'{prefix}/mlm-hidden/layernorm/gamma',
            ],
            'MLM-Matching': [f'{prefix}/mlm-matching/bias'],
        })
        return mapping

    def check_var_status(self, key: str, val: Any) -> bool:
        return self.__var_status.get(key) == val

    @staticmethod
    def fields_to_check():
        """ define fields to be check in export checkpoint
        """
        return ['bias', 'kernel', 'beta', 'gamma', 'embeddings', 'gau_kernel', 'encode_attn',
                'output_attn', 'output_layernorm', 'cell_initializer', 'cell_residual_layernorm',
                'cell_glu']

    @staticmethod
    def export_checkpoint(config_path: str, ckpt_path: str, target_path: str):
        """ Export the default checkpoint to a concise checkpoint removing
            redundant variables and assigning meaningful names.
        Args:
            config_path: str, default config path
            ckpt_path: str, default checkpoint path
            target_path: str, target checkpoint path
        """
        params = RanNetParams.from_file(config_path)
        with tf.compat.v1.Session() as sess:
            var_names = [var_name for var_name, _ in tf.train.list_variables(ckpt_path)
                         if "optimizer" not in var_name and "global_step" not in var_name and "_CHECKPOINTABLE_OBJECT_GRAPH" not in var_name]  # NOQA
            expected_names = list(itertools.chain(*[
                v for _, v in RanNet.variable_mapping(ran_layers=params.ran_layers).items()]))
            assert len(var_names) == len(expected_names), f"failed to export due to variable names size ({len(var_names)}) and expected names size ({len(expected_names)}) not matching."  # NOQA
            for var_name, expected_name in zip(var_names, expected_names):
                # check matching
                for name in RanNet.fields_to_check():
                    if name in var_name:
                        new_name = name.replace('_', '-')
                        new_expected_name = expected_name.replace('_', '-')
                        assert new_name in new_expected_name, f'{new_name} not in {new_expected_name}'
                var = tf.train.load_variable(ckpt_path, var_name)
                # set the new name
                var = tf.Variable(var, name=expected_name)
            # save the variables
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, target_path)

    def restore_weights_from_checkpoint(self,
                                        model: Models,
                                        checkpoint_path: str,
                                        variable_mapping: Optional[Dict] = None,
                                        ran_layers: int = 2):
        """ Restore weights from checkpoint
        Args:
            model: Models. Keras Model
            checkpoint_path: str. Path to checkpoint
            variable_mapping: Optional[Dict]. Variable mapping. Defaults to None, use the default mapping.
        """
        def get_var_name(layer_name: str, weight_name: str) -> str:
            weight_name = self.remove_prefix(weight_name.split(':')[0]).lower()
            if layer_name.lower() != weight_name.split('/')[0]:
                # compatible with TF2.5x
                weight_name = f'{layer_name.lower()}/{weight_name}'
            for key in ['/cell-initializer-group', '/window-loop']:
                weight_name = weight_name.replace(key, '')
            return weight_name

        variable_mapping = variable_mapping or RanNet.variable_mapping(ran_layers=ran_layers)
        variables = load_variables(checkpoint_path)
        for layer_name, _ in variable_mapping.items():
            layer_name = self.get_weight_name(layer_name)
            try:
                var_names = ['rannet/' + get_var_name(layer_name, weight.name)
                             for weight in model.get_layer(name=layer_name).weights]
                model.get_layer(name=layer_name).set_weights([
                    variables(var_name) for var_name in var_names
                ])
            except Exception as error:
                if 'MLM' in layer_name:
                    log.info(f'Skip {layer_name}')
                else:
                    raise ValueError(f'failed load layer {layer_name}, error: {error}')
        return model

    @staticmethod
    def load_rannet(config_path: str,
                    checkpoint_path: str,
                    window_size: Optional[int] = None,
                    with_mlm: bool = False,
                    with_cell: bool = False,
                    **kwargs) -> Tuple[object, Models]:
        """ Load pretrained RanNet model
        Args:
            config_path: str. Path to config
            checkpoint_path: str. Path to checkpoint
            with_mlm: bool. Wether to return mlm output. Defaults to False
            kwargs: Other kwargs of RanNet
        Return:
            RanNet: RanNet object
            model: RanNet keras model
        """
        params = RanNetParams.from_file(config_path)
        if window_size is not None:
            params.window_size = window_size
        rannet = RanNet(params, **kwargs)
        model = rannet(with_mlm=with_mlm, with_cell=with_cell)
        model = rannet.restore_weights_from_checkpoint(model, checkpoint_path, ran_layers=params.ran_layers)
        return rannet, model

    @staticmethod
    def compile(model: Models,
                learning_rate: float = 1e-3,
                weight_decay: float = 0.01,
                loss: Union[Dict, Callable, str] = 'sparse_categorical_crossentropy',
                lr_schedule: Optional[Dict[int, float]] = None,
                gradient_accumulation_steps: Optional[int] = None,
                **kwargs):
        model.summary()
        optim_kwargs = {}
        if TF_KERAS:
            optim_kwargs['bias_correction'] = False
        model.compile(
            optimizer=AdamWarmup(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                lr_schedule=lr_schedule,
                gradient_accumulation_steps=gradient_accumulation_steps,
                exclude_weight_decay_pattern=['LayerNorm', 'Norm', 'bias'],
                **optim_kwargs
            ),
            loss=loss,
            **kwargs
        )


class RanNetForLM(RanNet):
    def __init__(self,
                 params: RanNetParams,
                 return_cell: bool = False,
                 prefix: str = ''):
        super().__init__(params,
                         return_sequences=True,
                         return_cell=return_cell,
                         apply_cell_transform=False,
                         mlm_softmax=True,
                         apply_lm_mask=True,
                         apply_seq2seq_mask=False,
                         prefix=prefix)

    def __call__(self, with_cell: bool = False, return_model: bool = True) -> Union[Models, Tensors]:
        return super().__call__(with_cell=with_cell, with_mlm=True, return_model=return_model, seq2seq=False)


class RanNetForAdaptiveLM(RanNet):
    def __init__(self,
                 params: RanNetParams,
                 cutoffs: List[int],
                 div_val: int = 1,
                 output_dropout_rate: float = 0.0,
                 return_cell: bool = False,
                 prefix: str = ''):
        super().__init__(params,
                         return_sequences=True,
                         return_cell=return_cell,
                         mlm_softmax=False,
                         apply_cell_transform=False,
                         apply_lm_mask=True,
                         apply_seq2seq_mask=False,
                         prefix=prefix)
        dropout_rate = output_dropout_rate or self.params.dropout_rate
        self.output_dropout = L.Dropout(dropout_rate, name='Output-Dropout')
        self.token_embedding = AdaptiveEmbedding(
            input_dim=self.params.vocab_size,
            output_dim=self.params.embedding_size,
            cutoffs=cutoffs,
            div_val=div_val,
            return_embeddings=True,
            return_projections=True,
            mask_zero=False,
            name=self.get_weight_name('Adaptive-Embedding'),
        )
        self.adaptive_softmax = AdaptiveSoftmax(
            input_dim=self.params.embedding_size,
            output_dim=self.params.vocab_size,
            cutoffs=cutoffs,
            div_val=div_val,
            bind_embeddings=True,
            bind_projections=True,
            name=self.get_weight_name('Adaptive-Softmax'),
        )
        self.output_mapping = L.Dense(
            self.params.embedding_size,
            activation='swish',
            kernel_initializer=self.initializer,
            name=self.get_weight_name('Output-Mapping')
        )

    def __call__(self, with_cell: bool = False, return_model: bool = True) -> Union[Models, Tensors]:
        x, cell, _ = self.get_inputs(with_cell=with_cell, with_segment=False)
        x_mask = self.text_masking(x)

        emb = self.token_embedding(x)
        x, projections = emb[0], emb[1:]
        outputs = self.encode(x, x_mask, cell)

        cell = None
        if isinstance(outputs, (list, tuple)):
            outputs, cell = outputs[0], outputs[1]
        outputs = self.output_dropout(outputs)
        outputs = self.output_mapping(outputs)
        outputs = self.adaptive_softmax([outputs] + projections)

        outputs = [outputs, cell] if cell is not None else outputs
        if return_model:
            return keras.Model(self.inputs, outputs)
        return outputs


class RanNetForSeq2Seq(RanNet):
    def __init__(self,
                 params: RanNetParams,
                 prefix: str = ''):
        super().__init__(params,
                         return_sequences=True,
                         return_cell=False,
                         mlm_softmax=True,
                         apply_cell_transform=False,
                         apply_lm_mask=False,
                         apply_seq2seq_mask=True,
                         prefix=prefix)

    def __call__(self, with_cell: bool = False, return_model: bool = True) -> Union[Models, Tensors]:
        return super().__call__(with_cell=with_cell, with_mlm=True, return_model=return_model, seq2seq=True)


class RanNetForMLMPretrain(RanNet):
    def __init__(self, params: RanNetParams, **kwargs):
        super().__init__(params,
                         return_cell=False,
                         return_sequences=True,
                         apply_cell_transform=False,
                         **kwargs)

    def __call__(self) -> Tuple[Models, Models, Dict]:
        mlm_output = super().__call__(with_mlm=True, return_model=False)
        token_ids_in = L.Input(name=self.get_weight_name('token_ids'),
                               shape=(None,), dtype=K.floatx())  # ground-truth token ids
        mlm_mask_in = L.Input(name=self.get_weight_name('mlm_mask'), shape=(None,), dtype=K.floatx())  # mlm mask
        self.inputs += [token_ids_in, mlm_mask_in]
        token_ids, mlm_mask = token_ids_in, mlm_mask_in

        # with mlm loss
        mlm_output = WithSparseCategoricalCrossEntropy(
            name=self.get_weight_name('mlm_loss'))([token_ids, mlm_output, mlm_mask])
        # with mlm acc
        output = WithSparseCategoricalAccuracy(
            name=self.get_weight_name('mlm_acc'))([token_ids, mlm_output, mlm_mask])
        train_model = keras.Model(
            inputs=self.inputs, outputs=output
        )

        return train_model
