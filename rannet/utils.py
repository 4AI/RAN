# -*- coding: utf-8 -*-

from typing import Optional, Union
from langml import K
from langml.tensor_typing import Tensors


def triangular_causal_mask(seq_len: Union[int, Tensors]) -> Tensors:
    ''' Generate triangular causal mask
    Args:
        seq_len: sequence len
    Examples:
        for seq_len = 3, the mask is:
        array([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]], dtype=float32)
    '''
    idxs = K.arange(0, seq_len)
    mask = idxs[None, :] <= idxs[:, None]
    mask = K.cast(mask, K.floatx())
    return -(1 - mask) * 1e12


def prefix_causal_mask(segment: Tensors) -> Tensors:
    """ Generate prefix causal mask
    Args:
        segment: segment ids
    Examples:
        for segment [[0, 0, 0, 1, 1]], the mask is;
        array([[[1., 1., 1., 0., 0.],
                [1., 1., 1., 0., 0.],
                [1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1.]]], dtype=float32)
    """
    idxs = K.cumsum(segment, axis=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = K.cast(mask, K.floatx())
    return -(1 - mask) * 1e12


def standard_normalize(x: Tensors, epsilon: float = 1e-7) -> Tensors:
    mean = K.mean(x, axis=-1, keepdims=True)
    variance = K.mean(K.square(x), axis=-1, keepdims=True)
    std = K.sqrt(variance + epsilon)
    # standard normalization: x = (x - \mu) / \std
    return (x - mean) / std


def mean(x: Tensors, mask: Optional[Tensors] = None, axis: float = -1, keepdims: bool = False) -> Tensors:
    if mask is not None:
        mask_expand = K.expand_dims(mask, axis=2) if K.ndim(mask) == 2 else mask
        o = K.sum(x * mask_expand, axis=axis, keepdims=keepdims) / K.sum(mask_expand)
    else:
        o = K.mean(x, axis=axis, keepdims=keepdims)
    return o
