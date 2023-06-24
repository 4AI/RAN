# -*- coding: utf-8 -*-

""" Implementation of Recurrent Attention Network
"""

from typing import Dict, Optional, Tuple, List, Callable, Union

import tensorflow as tf
from langml import keras, L, K
from langml.layers import SineCosinePositionEmbedding, LayerNorm
from langml.tensor_typing import Tensors, Initializer, Activation

from rannet.utils import triangular_causal_mask, prefix_causal_mask, standard_normalize, mean


def align(tensor: Tensors, axes: int, ndim: Optional[int] = None) -> Tensors:
    assert len(axes) == K.ndim(tensor)
    assert ndim or min(axes) >= 0
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]


def sequence_masking(x: Tensors,
                     mask: Optional[Tensors] = None,
                     value: Union[str, float] = '-inf',
                     axis: Optional[int] = None) -> Tensors:
    """mask sequence
    Args:
        x: input tensor
        mask: mask of input tensor
    """
    if mask is None:
        return x
    if isinstance(value, str):
        assert value in ['-inf', 'inf'], 'if value is a str, please choose it from [`-inf`, `inf`]'
    x_dtype = K.dtype(x)
    if x_dtype == 'bool':
        x = K.cast(x, 'int32')
    if K.dtype(mask) != K.dtype(x):
        mask = K.cast(mask, K.dtype(x))
    if value == '-inf':
        value = -1e12
    elif value == 'inf':
        value = 1e12
    if axis is None:
        axis = 1
    elif axis < 0:
        axis = K.ndim(x) + axis
    assert axis > 0, 'axis must be greater than 0'
    mask = align(mask, [0, axis], K.ndim(x))
    value = K.cast(value, K.dtype(x))
    x = x * mask + value * (1 - mask)
    if x_dtype == 'bool':
        x = K.cast(x, 'bool')
    return x


def apply_rotary_position_embeddings(sinusoidal: Tensors, *tensors):
    """ apply RoPE
    modified from: https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py#L310
    """
    assert len(tensors) > 0, 'at least one input tensor'
    assert all([
        K.int_shape(tensor) == K.int_shape(tensors[0]) for tensor in tensors[1:]
    ]), 'all tensors must have the same shape'
    ndim = K.ndim(tensors[0])
    sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
    cos_pos = K.repeat_elements(sinusoidal[..., 1::2], 2, -1)
    sin_pos = K.repeat_elements(sinusoidal[..., ::2], 2, -1)
    outputs = []
    for tensor in tensors:
        tensor2 = K.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
        tensor2 = K.reshape(tensor2, K.shape(tensor))
        outputs.append(tensor * cos_pos + tensor2 * sin_pos)
    return outputs[0] if len(outputs) == 1 else outputs


class GatedLinearUnit(L.Layer):
    def __init__(self,
                 units: int,
                 kernel_initializer: Initializer = 'glorot_normal',
                 **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.supports_masking = True

    def get_config(self) -> dict:
        config = {
            "units": self.units,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GatedLinearUnit, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        super(GatedLinearUnit, self).build(input_shape)
        self.linear = L.Dense(self.units, kernel_initializer=self.kernel_initializer, name='dense-t')
        self.sigmoid = L.Dense(self.units, activation='sigmoid',
                               kernel_initializer=self.kernel_initializer, name='dense-g')

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None):
        return self.linear(inputs) * self.sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs: Tensors, mask: Optional[Tensors] = None):
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    @staticmethod
    def get_custom_objects() -> dict:
        return {'GatedLinearUnit': GatedLinearUnit}


class SelfAttention(L.Layer):
    def __init__(self,
                 units: Optional[int] = None,
                 return_attention: bool = False,
                 is_residual: bool = True,
                 activation: Optional[Activation] = None,
                 kernel_initializer: Initializer = 'glorot_normal',
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self.supports_masking = True

        self.units = units
        self.return_attention = return_attention
        self.is_residual = is_residual
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

    def get_config(self) -> dict:
        config = {
            "units": self.units,
            "return_attention": self.return_attention,
            "is_residual": self.is_residual,
            "activation": keras.activations.serialize(self.activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate
        }
        base_config = super(SelfAttention, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        super(SelfAttention, self).build(input_shape)

        if isinstance(input_shape, list):
            feature_dim = int(input_shape[-1][-1])
        else:
            feature_dim = int(input_shape[-1])

        units = feature_dim if self.units is None else self.units

        self.q_dense = L.Dense(
            units=units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-q',
        )
        self.k_dense = L.Dense(
            units=units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-k',
        )
        self.v_dense = L.Dense(
            units=units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-v',
        )
        self.o_dense = L.Dense(
            units=feature_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-o',
        )
        if self.is_residual:
            self.glu = GatedLinearUnit(
                feature_dim, kernel_initializer=self.kernel_initializer, name='glu')
            self.layernorm = LayerNorm(name='layernorm')

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Union[List[Tensors], Tensors]:
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs

        mask = mask[0] if isinstance(mask, list) else mask

        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)

        e = K.batch_dot(qw, kw, axes=2)
        e /= K.int_shape(qw)[-1]**0.5
        # axis=1 if channel_last else 2
        e = sequence_masking(e, mask, '-inf', 1)
        a = K.softmax(e)
        if self.dropout_rate:
            a = L.Dropout(self.dropout_rate)(a)
        out = K.batch_dot(a, vw)
        out = self.o_dense(out)
        if self.is_residual:
            # out += qw
            if self.dropout_rate > 0:
                out = L.Dropout(self.dropout_rate)(out)
            out = out + self.glu(qw)
            out = self.layernorm(out)
        if self.return_attention:
            return [out, a]
        return out

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[List[Union[Tensors, None]], Tensors]:
        mask = mask[0] if isinstance(mask, list) else mask
        if self.return_attention:
            return [mask, None]
        return mask

    def compute_output_shape(self, input_shape: Tensors) -> Union[List[Tensors], Tensors]:
        if not isinstance(input_shape, list):
            output_shape = input_shape[0]
        else:
            output_shape = input_shape
        if self.return_attention:
            attention_shape = (output_shape[0], output_shape[1], output_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    @staticmethod
    def get_custom_objects() -> dict:
        return {'SelfAttention': SelfAttention}


class PosMultiHeadAttention(L.Layer):
    def __init__(
        self,
        head_num,
        head_size,
        use_bias=True,
        dropout_rate=None,
        return_attention=False,
        activation=None,
        kernel_initializer='glorot_uniform',
        apply_relative_position=True,
        is_residual=True,
        **kwargs
    ):
        super(PosMultiHeadAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.head_size = head_size
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.return_attention = return_attention
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.apply_relative_position = apply_relative_position
        self.is_residual = is_residual

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'head_size': self.head_size,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
            'return_attention': self.return_attention,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'apply_relative_position': self.apply_relative_position,
            'is_residual': self.is_residual,
        }
        base_config = super(PosMultiHeadAttention, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape):
        super(PosMultiHeadAttention, self).build(input_shape)

        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        feature_dim = int(input_shape[0][-1])
        self.q_dense = L.Dense(
            units=self.head_size * self.head_num,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-q',
        )
        self.k_dense = L.Dense(
            units=self.head_size * self.head_num,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-k',
        )
        self.v_dense = L.Dense(
            units=self.head_size * self.head_num,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-v',
        )
        self.o_dense = L.Dense(
            units=feature_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-o',
        )
        if self.is_residual:
            self.glu = GatedLinearUnit(
                feature_dim, kernel_initializer=self.kernel_initializer, name='glu')
            self.layernorm = LayerNorm(name='layernorm')

    def call(self, inputs, mask=None, attn_bias=None, **kwargs):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask

        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)

        batch_size = K.shape(qw)[0]
        qw = K.reshape(qw, (batch_size, -1, self.head_num, self.head_size))
        kw = K.reshape(kw, (batch_size, -1, self.head_num, self.head_size))
        vw = K.reshape(vw, (batch_size, -1, self.head_num, self.head_size))

        o, attn = self.compute_attention(
            [qw, kw, vw],
            mask=[q_mask, k_mask, v_mask],
            attn_bias=attn_bias,
        )

        shape, int_shape = K.shape(o), K.int_shape(o)
        o = K.reshape(o, (shape[0], shape[1], int_shape[2] * int_shape[3]))  # (B, L, N * ND)
        o = self.o_dense(o)
        if self.is_residual:
            # residual with raw input (q)
            if self.dropout_rate > 0:
                o = L.Dropout(self.dropout_rate)(o)
            o = o + self.glu(q)
            o = self.layernorm(o)
        if self.return_attention:
            return [o, attn]
        return o

    def compute_attention(self, inputs, mask=None, attn_bias=None):
        qw, kw, vw = inputs
        if mask is not None:
            _, _, v_mask = mask
        else:
            v_mask = None
        # apply
        if self.apply_relative_position:
            pos = SineCosinePositionEmbedding("zero", output_dim=self.head_size)(kw)
            qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
        # Attention
        e = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        e /= self.head_size**0.5
        if attn_bias is not None:
            if K.ndim(attn_bias) == 3:
                attn_bias = align(attn_bias, [0, -2, -1], K.ndim(e))
            e += attn_bias
        e = sequence_masking(e, v_mask, '-inf', -1)
        attn = K.softmax(e)
        if self.dropout_rate:
            attn = L.Dropout(self.dropout_rate)(attn)
        o = tf.einsum('bhjk,bkhd->bjhd', attn, vw)
        return o, attn

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q_shape, _, v_shape = input_shape
        else:
            q_shape = _ = v_shape = input_shape
        output_shape = q_shape[:-1] + (v_shape[-1],)
        if self.return_attention:
            attention_shape = (q_shape[0], self.head_num, q_shape[1], v_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    @staticmethod
    def get_custom_objects() -> dict:
        return {'PosMultiHeadAttention': PosMultiHeadAttention}


def ran(inputs: Tensors,
        encode_attn: PosMultiHeadAttention,
        cell: Optional[Tensors] = None,
        segments: Optional[Tensors] = None,
        cell_initializer_type: str = 'zero',
        cell_initializer: Optional[Callable] = None,
        cell_glu: Optional[Callable] = None,
        cell_residual_layernorm: Optional[Callable] = None,
        mask: Optional[Tensors] = None,
        apply_lm_mask: bool = False,
        apply_seq2seq_mask: bool = False,
        window_size: int = 128,
        concat_layernorm=None,
        memory_review=None,
        dropout_rate: float = 0.0,
        min_window_size: int = 16,
        cell_pooling: str = 'last'):
    """ Core implementation
    """

    def do_step(cell, current_input, mask=None, segment=None):
        current_input = K.concatenate((cell, current_input), axis=1)  # (B, 1 + W, D)
        # avoid gradient explosion
        # x' = \gamma * \frac{x - \mu}{\sigma} + \beta
        current_input = concat_layernorm(current_input)
        if dropout_rate > 0:
            current_input = L.Dropout(dropout_rate)(current_input)

        attn_mask = None
        if mask is not None:
            cell_mask = K.ones_like(mask[:, :1])
            attn_mask = K.concatenate((cell_mask, mask), axis=1)

        attn_bias = None
        if apply_seq2seq_mask:
            # apply prefix causal mask (unilm mask)
            cell_segment = K.zeros_like(segment[:, :1])
            segment = K.concatenate((cell_segment, segment), axis=1)
            attn_bias = prefix_causal_mask(segment)
        elif apply_lm_mask:
            # apply causal mask
            attn_bias = triangular_causal_mask(K.shape(current_input)[1])

        # encoder
        output = encode_attn(current_input, mask=attn_mask, attn_bias=attn_bias)
        cell, output = output[:, 0], output[:, 1:]
        cell = K.expand_dims(cell, axis=1)  # (B, 1, D)

        # normalize output and cell seperately
        output = standard_normalize(output)
        if dropout_rate > 0:
            output = L.Dropout(dropout_rate)(output)

        cell = standard_normalize(cell)
        if dropout_rate > 0:
            cell = L.Dropout(dropout_rate)(cell)

        return cell, output

    def do_first_step(counter, ranges, range_size, cell=None, mask=None):
        ind = tf.cond(
            range_size > 0,
            true_fn=lambda: K.gather(ranges, counter),  # time_steps > window_size
            false_fn=lambda: K.arange(
                0,
                K.shape(inputs)[1] + 1, K.shape(inputs)[1],
                dtype='int32'),  # [0, K.shape(inputs)[1]], time_steps < window_size
        )

        # init cell
        if cell is None:
            if cell_initializer_type == 'mean':
                # initialize cell with mean pooling
                cell = mean(inputs, mask, axis=1, keepdims=True)
            elif cell_initializer_type == 'zero':
                # initialize cell with zeros
                cell = K.zeros_like(inputs[:, 0])
                cell = K.expand_dims(cell, axis=1)
            else:
                raise ValueError(f'invalid assignment {cell_initializer_type}, '
                                 'please specify cell_initializer_type from [`mean`, `zero`]')
        else:
            if K.ndim(cell) == 2:
                cell = K.expand_dims(cell, axis=1)
        # with weight
        cell = cell_initializer(cell)
        if dropout_rate > 0:
            cell = L.Dropout(dropout_rate)(cell)

        # slice window input
        current_input = inputs[:, ind[0]: ind[1]]  # (B, W, D)
        current_mask = mask[:, ind[0]: ind[1]] if mask is not None else None
        current_segment = segments[:, ind[0]: ind[1]] if segments is not None else None

        # encode
        cell, output = do_step(cell, current_input, mask=current_mask, segment=current_segment)

        cells = cell
        prev_cell = cell
        return counter + 1, cell, prev_cell, output, cells

    def do_window_step(counter, ranges, range_size, cell, prev_cell=None, outputs=None, cells=None, mask=None):
        ind = K.gather(ranges, counter)
        current_input = inputs[:, ind[0]: ind[1]]  # (B, W, D)
        current_mask = mask[:, ind[0]: ind[1]] if mask is not None else None
        current_segment = segments[:, ind[0]: ind[1]] if segments is not None else None

        cell, current_output = do_step(cell, current_input, mask=current_mask, segment=current_segment)
        outputs = K.concatenate((outputs, current_output), axis=1)
        cells = K.concatenate((cells, cell), axis=1)

        # cell with residual
        cell_t = cell
        # cell += prev_cell
        cell = cell + cell_glu(prev_cell)
        cell = cell_residual_layernorm(cell)
        if dropout_rate > 0:
            cell = L.Dropout(dropout_rate)(cell)
        prev_cell = cell_t
        return counter + 1, ranges, range_size, cell, prev_cell, outputs, cells, mask

    def do_final_step(last_ind, cell, prev_cell, outputs, cells, mask=None):
        current_input = inputs[:, last_ind[1]:]
        current_mask = mask[:, last_ind[1]:]
        current_segment = segments[:, last_ind[1]:] if segments is not None else None
        cell, current_output = do_step(cell, current_input, mask=current_mask, segment=current_segment)
        outputs = K.concatenate((outputs, current_output), axis=1)
        cells = K.concatenate((cells, cell), axis=1)

        # cell with residual
        # cell += prev_cell
        cell = cell + cell_glu(prev_cell)
        cell = cell_residual_layernorm(cell)
        return cell, outputs, cells

    def rearrange_ranges(ranges, range_size, seq_len):
        # if ranges is empty, ranges[-1] will raise error
        last_range = K.gather(ranges, range_size - 1)
        start = last_range[0]
        end = last_range[1]
        final_len = seq_len - end
        ranges, flag = tf.cond(
            tf.logical_and(end > start, final_len < min_window_size),
            true_fn=lambda: (
                K.concatenate((ranges[:-1], K.reshape(K.stack((start, seq_len + 1)), (1, -1))), axis=0),
                K.constant(True, dtype='bool')),
            false_fn=lambda: (ranges, K.constant(False, dtype='bool'))
        )
        # new range size
        range_size = K.shape(ranges)[0]
        return ranges, range_size, flag

    seq_len = K.shape(inputs)[1]
    ranges = K.arange(0, seq_len, window_size)
    starts = K.expand_dims(ranges[0:-1:], axis=1)
    ends = K.expand_dims(ranges[1::], axis=1)
    ranges = K.concatenate([starts, ends], axis=-1)
    range_size = K.shape(ranges)[0]

    # rearrange ranges to ensure the step size of the final-step is greater than min_window_size
    rearrange_flag = K.constant(False, dtype='bool')
    ranges, range_size, rearrange_flag = tf.cond(
        range_size > 0,
        true_fn=lambda: rearrange_ranges(ranges, range_size, seq_len),
        false_fn=lambda: (ranges, range_size, rearrange_flag)
    )

    counter = K.constant(0, dtype='int32', name="counter")
    # first step
    counter, cell, prev_cell, outputs, cells = do_first_step(counter, ranges, range_size, cell=cell, mask=mask)
    # window loop
    loop_outputs = tf.while_loop(
        cond=lambda counter, *_: counter < range_size,
        body=do_window_step,
        loop_vars=[counter, ranges, range_size, cell, prev_cell, outputs, cells, mask],
        shape_invariants=[
            counter.get_shape(), ranges.get_shape(), range_size.get_shape(), cell.get_shape(), prev_cell.get_shape(),
            outputs.get_shape(), tf.TensorShape([None, None, K.int_shape(inputs)[-1]]), mask.get_shape()],
        name='window-loop',
    )
    cell, prev_cell, outputs, cells = loop_outputs[3], loop_outputs[4], loop_outputs[5], loop_outputs[6]
    # final step
    cell, outputs, cells = tf.cond(
        tf.logical_and(range_size > 0, tf.logical_not(rearrange_flag)),
        true_fn=lambda: do_final_step(K.gather(ranges, range_size - 1), cell, prev_cell, outputs, cells, mask=mask),
        false_fn=lambda: (cell, outputs, cells),
        name='final-step'
    )

    if isinstance(memory_review, SelfAttention):
        outputs = memory_review([outputs, cells, cells], mask=mask)
    else:
        outputs = memory_review(outputs)
    if cell_pooling == 'last':
        cell = K.squeeze(cell, axis=1)  # (B, D)
    elif cell_pooling == 'mean':
        cell = K.mean(cells, axis=1)
    elif cell_pooling == 'max':
        cell = K.max(cells, axis=1)
    else:
        raise ValueError('Please specify `cell_pooling` from [`last`, `mean`, `max`]')
    return outputs, cell


class RAN(L.Layer):
    def __init__(self,
                 head_num: int,
                 head_size: int = 256,
                 window_size: int = 128,
                 min_window_size: int = 16,
                 activation: Activation = 'swish',
                 kernel_initializer: Initializer = 'glorot_normal',
                 apply_lm_mask: bool = False,
                 apply_seq2seq_mask: bool = False,
                 apply_memory_review: bool = True,
                 dropout_rate: float = 0.0,
                 cell_initializer_type: str = 'zero',
                 cell_pooling: str = 'last',
                 **kwargs):
        super(RAN, self).__init__(**kwargs)
        assert window_size > min_window_size, "window_size must be greater than min_window_size"
        self.head_num = head_num
        self.head_size = head_size
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.apply_lm_mask = apply_lm_mask
        self.apply_seq2seq_mask = apply_seq2seq_mask
        self.apply_memory_review = apply_memory_review
        self.dropout_rate = dropout_rate
        self.cell_initializer_type = cell_initializer_type
        self.cell_pooling = cell_pooling
        self.supports_masking = True

    def get_config(self):
        config = {
            "head_num": self.head_num,
            "head_size": self.head_size,
            "window_size": self.window_size,
            "min_window_size": self.min_window_size,
            "activation": keras.activations.serialize(self.activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "apply_lm_mask": self.apply_lm_mask,
            "apply_seq2seq_mask": self.apply_seq2seq_mask,
            "apply_memory_review": self.apply_memory_review,
            "dropout_rate": self.dropout_rate,
            "cell_initializer_type": self.cell_initializer_type,
            "cell_pooling": self.cell_pooling
        }
        base_config = super(RAN, self).get_config()
        return dict(base_config, **config)

    def build(self, input_shape: Union[Tensors, List[Tensors]]):
        super(RAN, self).build(input_shape)
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self.encode_attn = PosMultiHeadAttention(
            head_num=self.head_num,
            head_size=self.head_size,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            apply_relative_position=True,
            is_residual=True,
            kernel_initializer=self.kernel_initializer,
            name='encode-attn')
        self.cell_initializer = keras.Sequential([
            L.Dense(
                input_shape[0][-1],
                activation=self.activation,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                name='cell-initializer'
            ),
            LayerNorm(name='cell-initializer-layernorm')
        ], name='cell-initializer-group')
        self.cell_glu = GatedLinearUnit(
            input_shape[0][-1], kernel_initializer=self.kernel_initializer, name='cell-glu')
        self.cell_residual_layernorm = LayerNorm(name='cell-residual-layernorm')
        self.concat_layernorm = LayerNorm(name='concat-layernorm')

        if self.apply_memory_review:
            # apply memory review mechanism
            self.memory_review = SelfAttention(
                activation=self.activation,
                is_residual=True,
                kernel_initializer=self.kernel_initializer,
                dropout_rate=self.dropout_rate,
                name='output-attn'
            )
        else:
            # apply layernorm
            self.memory_review = LayerNorm(name='output-layernorm')

    def compute_mask(self,
                     inputs: Union[Tensors, List[Tensors]],
                     mask: Optional[List[Tensors]] = None) -> Union[Tuple[Tensors, Tensors], Tensors]:
        if isinstance(inputs, list):
            mask = inputs[1]
        mask = mask[0] if isinstance(mask, list) else mask
        return [mask, None]

    def call(self,
             inputs: Union[Tensors, List[Tensors]],
             mask: Optional[List[Tensors]] = None,
             cell: Optional[Tensors] = None,
             segments: Optional[Tensors] = None) -> Tuple[Tensors, Tensors]:
        if isinstance(inputs, list):
            inputs, mask = inputs[0], inputs[1]

        # text mask
        mask = mask[0] if isinstance(mask, list) else mask
        assert mask is not None, "mask should not be None!"

        outputs, cell = ran(
            inputs,
            self.encode_attn,
            cell=cell,
            segments=segments,
            cell_initializer_type=self.cell_initializer_type,
            cell_initializer=self.cell_initializer,
            cell_glu=self.cell_glu,
            cell_residual_layernorm=self.cell_residual_layernorm,
            mask=mask,
            apply_lm_mask=self.apply_lm_mask,
            apply_seq2seq_mask=self.apply_seq2seq_mask,
            window_size=self.window_size,
            concat_layernorm=self.concat_layernorm,
            memory_review=self.memory_review,
            dropout_rate=self.dropout_rate,
            min_window_size=self.min_window_size,
            cell_pooling=self.cell_pooling
        )
        return [outputs, cell]

    def compute_output_shape(self,
                             input_shape: Union[Tensors, List[Tensors]]) -> Union[Tensors, Tuple[Tensors, Tensors]]:
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        cell_shape = (input_shape[0][0], input_shape[0][-1])
        return [input_shape[0], cell_shape]

    @staticmethod
    def get_custom_objects() -> Dict:
        return {'RAN': RAN}


custom_objects = {}
custom_objects.update(SelfAttention.get_custom_objects())
custom_objects.update(PosMultiHeadAttention.get_custom_objects())
custom_objects.update(GatedLinearUnit.get_custom_objects())
custom_objects.update(RAN.get_custom_objects())
keras.utils.get_custom_objects().update(custom_objects)
