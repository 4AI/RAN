# -*- coding: utf-8 -*-

import re
from typing import List, Optional, Dict

import tensorflow as tf
from langml import keras, K, TF_KERAS
from langml.log import warn


symbolic = lambda f: f  # NOQA
K.symbolic = getattr(K, 'symbolic', None) or symbolic


def piecewise_linear(t: int, schedule: Dict[int, float], from_zero: bool = True):
    """piecewise linear
    modified from:
        https://github.com/bojone/bert4keras/blob/9c1c916def4d515a046c414c9849b2e7e11af1e3/bert4keras/backend.py#L73

    Args:
        t: int, iterations
        schedule: Dict[int, float], e.g., for {1000: 1, 2000: 0.1},
            when t ∈ [0, 1000], ratio increase from 0.0 to 1.0 uniformly,
            when t ∈ [1000, 2000], ratio decrease from 1.0 to 0.1 evenly,
            when t > 2000, ratio keep 0.1
    """
    schedule = sorted(schedule.items())
    if from_zero and schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    t = K.cast(t, K.floatx())
    x = (t * 0 + 1) * schedule[0][1]
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = (t * 0 + 1) * schedule[i][1]
        x = K.switch(t >= t_begin, x, x_begin)

    return x


class AdamWarmup(keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 amsgrad: bool = False,
                 decay: float = 0.0,
                 weight_decay: float = 0.0,
                 epsilon: float = 1e-7,
                 lr_schedule: Optional[Dict[int, float]] = None,
                 gradient_accumulation_steps: int = None,
                 exclude_weight_decay_pattern: Optional[List[str]] = None,
                 include_weight_decay_pattern: Optional[List[str]] = None,
                 **kwargs):
        """Adam optimizer with warmup setting.
        Modified from: https://github.com/keras-team/keras/blob/2.3.1/keras/optimizers.py#L467

        Default parameters follow those provided in the original paper.
        # Arguments
            learning_rate: float >= 0. Learning rate.
            beta_1: float, 0 < beta < 1. Generally close to 1.
            beta_2: float, 0 < beta < 1. Generally close to 1.
            amsgrad: boolean. Whether to apply the AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                Beyond".
            decay: float > 0. learning rate decay.
            weight_decay: float > 0. weight decay.
            epsilon: float. If `None`, defaults to `K.epsilon()`.
        # References
            - [Adam - A Method for Stochastic Optimization](
            https://arxiv.org/abs/1412.6980v8)
            - [On the Convergence of Adam and Beyond](
            https://openreview.net/forum?id=ryQu7f-RZ)
        """
        self.initial_decay = decay
        self.epsilon = epsilon if epsilon is not None else K.epsilon()
        learning_rate = kwargs.pop('lr', learning_rate)
        super(AdamWarmup, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.exclude_weight_decay_pattern = exclude_weight_decay_pattern
        self.include_weight_decay_pattern = include_weight_decay_pattern
        if self.exclude_weight_decay_pattern is not None and self.include_weight_decay_pattern is not None:
            warn('if both `exclude_weight_decay_pattern` and `include_weight_decay_pattern` are set, '
                 'only `exclude_weight_decay_pattern` will work.')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.gradient_accumulation_steps is not None:
            self.accum_grads = {}

    def _get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='v_' + str(i))
              for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p),
                     dtype=K.dtype(p),
                     name='vhat_' + str(i))
                     for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i))
                     for i in range(len(params))]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    @K.symbolic
    def get_updates(self, loss, params):
        if self.gradient_accumulation_steps is not None:
            grad_accum_cond = K.equal(self.iterations % self.gradient_accumulation_steps, 0)
            grad_accum_cond = K.cast(grad_accum_cond, K.floatx())

        if self.lr_schedule is not None:
            lr_multiplier = piecewise_linear(self.iterations, self.lr_schedule)

        old_update = K.update

        def new_update(x, new_x):
            if self.gradient_accumulation_steps is not None:
                new_x = grad_accum_cond * new_x + (1 - grad_accum_cond) * x

            if any(x is p for p in params):
                # do lr schedule
                if self.lr_schedule is not None:
                    new_x = x + (new_x - x) * lr_multiplier
                # do weight decay
                if self._handle_weight_decay_pattern(x):
                    new_x = new_x - self.learning_rate * self.weight_decay * x
            return old_update(x, new_x)

        K.update = new_update
        updates = self._get_updates(loss, params)
        K.update = old_update

        if self.gradient_accumulation_steps is not None:
            # get gradients
            grads = super(AdamWarmup, self).get_gradients(loss, params)
            accum_grads = [self.accum_grads[p] for p in params]
            # accumulate gradient
            with tf.control_dependencies(updates):
                accum_updates = [
                    K.update(ag, g + (1 - grad_accum_cond) * ag)
                    for g, ag in zip(grads, accum_grads)
                ]

            return accum_updates
        return updates

    def _handle_weight_decay_pattern(self, w):
        if self.exclude_weight_decay_pattern is not None:
            pattern = '|'.join(self.exclude_weight_decay_pattern)
            return not re.search(rf'({pattern})', w.name)
        if self.include_weight_decay_pattern is not None:
            pattern = '|'.join(self.include_weight_decay_pattern)
            return re.search(rf'({pattern})', w.name)
        return True

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'weight_decay': self.weight_decay,
                  'lr_schedule': self.lr_schedule,
                  'gradient_accumulation_steps': self.gradient_accumulation_steps,
                  'exclude_weight_decay_pattern': self.exclude_weight_decay_pattern,
                  'include_weight_decay_pattern': self.include_weight_decay_pattern}
        base_config = super(AdamWarmup, self).get_config()
        return dict(base_config, **config)

    @staticmethod
    def get_custom_objects():
        return {'AdamWarmup': AdamWarmup}


class AdamWarmupTF(tf.keras.optimizers.Optimizer):
    """
    tf keras adam warmup
    Modified from: https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L14
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        weight_decay: float = 0.,
        bias_correction: float = True,
        lr_schedule: Optional[Dict[int, float]] = None,
        gradient_accumulation_steps: int = None,
        exclude_weight_decay_pattern: Optional[List[str]] = None,
        include_weight_decay_pattern: Optional[List[str]] = None,
        name: str = 'AdamWarmupTF',
        **kwargs
    ):
        """
        # Arguments:
            learning_rate: float > 0, learning rate.
            beta_1: float, 0 < beta < 1. Generally close to 1.
            beta_2: float, 0 < beta < 1. Generally close to 1.
            epsilon: float. If `None`, defaults to `K.epsilon()`.
            weight_decay: float > 0. Weight decay.
            bias_correction: bool. Correct bias.
            lr_schedule: Optional[Dict[int, float]], learning rate scheduel,
                e.g. {1000: 1, 2000: 0.1, ...}. Defaults to None
            gradient_accumulation_steps: Optional[int], gradient accumulation steps. Defaults to None.
            include_weight_decay_pattern: list of str. The substring of weight names to be decayed.
            exclude_weight_decay_pattern: list of str. The substring of weight names to not be decayed.
        """
        kwargs['name'] = name
        super(AdamWarmupTF, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.weight_decay = weight_decay
        self.epsilon = epsilon or K.epislon()
        self.bias_correction = bias_correction
        self.lr_schedule = lr_schedule
        self.exclude_weight_decay_pattern = exclude_weight_decay_pattern
        self.include_weight_decay_pattern = include_weight_decay_pattern
        if self.exclude_weight_decay_pattern is not None and self.include_weight_decay_pattern is not None:
            warn('if both `exclude_weight_decay_pattern` and `include_weight_decay_pattern` are set, '
                 'only `exclude_weight_decay_pattern` will work.')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.gradient_accumulation_steps is not None:
            self.accum_grads = {}

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            if self.gradient_accumulation_steps is not None:
                self.add_slot(var, 'ag')

    def _do_resource_apply(self, grad, var, indices=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = K.pow(beta_1_t, local_step)
        beta_2_t_power = K.pow(beta_2_t, local_step)

        # do update
        if indices is None:
            m_t = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = K.update(v, beta_2_t * v + (1 - beta_2_t) * grad**2)
        else:
            mv_ops = [K.update(m, beta_1_t * m), K.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(
                    m, indices, (1 - beta_1_t) * grad
                )
                v_t = self._resource_scatter_add(
                    v, indices, (1 - beta_2_t) * grad**2
                )

        with tf.control_dependencies([m_t, v_t]):
            if self.bias_correction:
                m_t = m_t / (1.0 - beta_1_t_power)
                v_t = v_t / (1.0 - beta_2_t_power)
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            return K.update(var, var_t)

    def _decayed_lr(self, var_dtype):
        if self.lr_schedule is not None:
            lr_multiplier = piecewise_linear(self.iterations, self.lr_schedule)
            lr_t = super(AdamWarmupTF, self)._decayed_lr(var_dtype)
            return lr_t * K.cast(lr_multiplier, var_dtype)
        return super(AdamWarmupTF, self)._decayed_lr(var_dtype)

    def _resource_apply(self, grad, var, indices=None):
        if self.gradient_accumulation_steps is not None:
            grad_accum_cond = K.equal(self.iterations % self.gradient_accumulation_steps, 0)
            # get gradients
            ag = self.get_slot(var, 'ag')

        old_update = K.update

        def new_update(x, new_x):
            if self.gradient_accumulation_steps is not None:
                new_x = K.switch(grad_accum_cond, new_x, x)

            if x is var and self._handle_weight_decay_pattern(x):
                lr_t = self._decayed_lr(x.dtype.base_dtype)
                new_x = new_x - lr_t * self.weight_decay * x
            return old_update(x, new_x)

        K.update = new_update
        if self.gradient_accumulation_steps is not None:
            ag_t = ag / self.gradient_accumulation_steps
            op = self._do_resource_apply(ag_t, var)
        else:
            op = self._do_resource_apply(grad, var, indices)
        K.update = old_update

        if self.gradient_accumulation_steps is not None:
            # 累积梯度
            with tf.control_dependencies([op]):
                ag_t = K.switch(grad_accum_cond, K.zeros_like(ag), ag)
                with tf.control_dependencies([K.update(ag, ag_t)]):
                    if indices is None:
                        ag_t = K.update(ag, ag + grad)
                    else:
                        ag_t = self._resource_scatter_add(ag, indices, grad)
            return ag_t
        return op

    def _handle_weight_decay_pattern(self, w):
        if self.exclude_weight_decay_pattern is not None:
            pattern = '|'.join(self.exclude_weight_decay_pattern)
            return not re.search(rf'({pattern})', w.name)
        if self.include_weight_decay_pattern is not None:
            pattern = '|'.join(self.include_weight_decay_pattern)
            return re.search(rf'({pattern})', w.name)
        return True

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'lr_schedule': self.lr_schedule,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'exclude_weight_decay_pattern': self.exclude_weight_decay_pattern,
            'include_weight_decay_pattern': self.include_weight_decay_pattern,
            'bias_correction': self.bias_correction,
        }
        base_config = super(AdamWarmupTF, self).get_config()
        return dict(base_config, **config)

    @staticmethod
    def get_custom_objects():
        return {'AdamWarmupTF': AdamWarmupTF}


if TF_KERAS:
    AdamWarmup = AdamWarmupTF  # NOQA

custom_objects = {}
custom_objects.update(AdamWarmup.get_custom_objects())
keras.utils.get_custom_objects().update(custom_objects)
