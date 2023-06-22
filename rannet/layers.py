# -*- coding: utf-8 -*-


import tensorflow as tf
from langml import keras, K, L
from langml.tensor_typing import Tensors


def swish(x: Tensors) -> Tensors:
    return tf.nn.swish(x)


class AdaptiveEmbedding(L.Layer):
    # Credit: to adapt to Keras/TF Keras, this code is modified from: https://github.com/CyberZHG/keras-adaptive-softmax

    """Turns positive integers (indexes) into dense vectors of fixed size.
    # Arguments
        input_dim: int > 0. Size of the vocabulary.
        output_dim: int > 0. Dimension of the dense embedding after projection if it is not equal to embed_dim.
        embed_dim: int > 0. Dimension of the dense embedding.
        cutoffs: list of ints. Indices of splitting points.
        div_val: int >= 0. The scaling parameter of embedding.
        force_projection: Boolean. Add projection even if output_dim equals to embed_dim.
        embeddings_initializer: Initializer for the `embeddings` matrix.
        embeddings_regularizer: Regularizer function applied to the `embeddings` matrix.
        embeddings_constraint: Constraint function applied to the `embeddings` matrix.
        mask_zero: Whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using [recurrent layers](recurrent.md)
            which may take variable length input.
            If this is `True` then all subsequent layers
            in the model need to support masking or an exception will be raised.
            If mask_zero is set to True, as a consequence, index 0 cannot be
            used in the vocabulary (input_dim should equal size of
            vocabulary + 1).
    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    # References
        - [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf)
    """

    def __init__(self, input_dim, output_dim, embed_dim=None,
                 cutoffs=None, div_val=1,
                 force_projection=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 mask_zero=False,
                 return_embeddings=False,
                 return_projections=False,
                 **kwargs):
        super(AdaptiveEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if embed_dim is None:
            self.embed_dim = output_dim
        self.cutoffs = cutoffs
        if cutoffs is not None:
            if self.cutoffs[0] != 0:
                self.cutoffs = [0] + self.cutoffs
            if self.cutoffs[-1] != input_dim:
                self.cutoffs.append(input_dim)
        self.div_val = div_val
        self.force_projection = force_projection
        if force_projection is None:
            if div_val == 1:
                self.force_projection = False
            else:
                self.force_projection = True

        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)

        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.return_embeddings = return_embeddings
        self.return_projections = return_projections

        self.embeddings = None
        self.projections = None

    def build(self, input_shape):
        if self.div_val == 1:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.embed_dim),
                initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                name='embeddings',
            )
            if self.embed_dim != self.output_dim or self.force_projection:
                self.projections = self.add_weight(
                    shape=(self.embed_dim, self.output_dim),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='kernel',
                )
        else:
            self.embeddings, self.projections = [], []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                self.embeddings.append(self.add_weight(
                    shape=(self.cutoffs[i + 1] - self.cutoffs[i], embed_dim),
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint,
                    name='embeddings-{}'.format(i),
                ))
                projection_shape = (embed_dim, self.output_dim)
                if embed_dim == self.output_dim and not self.force_projection:
                    projection_shape = ()
                self.projections.append(self.add_weight(
                    shape=projection_shape,
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='kernel-{}'.format(i),
                ))
        super(AdaptiveEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            output_mask = None
        else:
            output_mask = K.not_equal(inputs, 0)
        if self.return_embeddings or self.return_projections:
            output_mask = [output_mask]
        if self.return_embeddings:
            if self.div_val == 1:
                output_mask += [None]
            else:
                output_mask += [None] * len(self.embeddings)
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    output_mask += [None]
            else:
                output_mask += [None] * len(self.projections)
        return output_mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape + (self.output_dim,)
        if self.return_embeddings or self.return_projections:
            output_shape = [output_shape]
        if self.return_embeddings:
            if self.div_val == 1:
                output_shape += [K.int_shape(self.embeddings)]
            else:
                output_shape += [K.int_shape(embed) for embed in self.embeddings]
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    output_shape += [K.int_shape(self.projections)]
            else:
                output_shape += [K.int_shape(proj) for proj in self.projections]
        return output_shape

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        if self.div_val == 1:
            out = K.gather(self.embeddings, inputs)
            if self.embed_dim != self.output_dim or self.force_projection:
                out = K.dot(out, self.projections)
        else:
            out = K.tile(
                K.expand_dims(K.zeros_like(inputs, dtype=K.floatx()), axis=-1),
                (1,) * K.ndim(inputs) + (self.output_dim,),
            )
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                low, high = self.cutoffs[i], self.cutoffs[i + 1]
                mask = K.cast(low <= inputs, K.floatx()) * K.cast(inputs < high, K.floatx())
                selected = K.gather(self.embeddings[i], (inputs - low) * K.cast(mask, 'int32'))
                if embed_dim != self.output_dim or self.force_projection:
                    projected = K.dot(selected, self.projections[i])
                else:
                    projected = selected
                out += projected * K.expand_dims(mask, axis=-1)
        if self.return_embeddings or self.return_projections:
            out = [out]
        if self.return_embeddings:
            if self.div_val == 1:
                out += [self.embeddings]
            else:
                out += [embed + 0.0 for embed in self.embeddings]
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    out += [self.projections]
            else:
                out += [proj + 0.0 for proj in self.projections]
        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'cutoffs': self.cutoffs,
            'div_val': self.div_val,
            'force_projection': self.force_projection,
            'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'mask_zero': self.mask_zero,
            'return_embeddings': self.return_embeddings,
            'return_projections': self.return_projections,
        }
        base_config = super(AdaptiveEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_custom_objects():
        return {'AdaptiveEmbedding': AdaptiveEmbedding}


class AdaptiveSoftmax(L.Layer):
    # Credit: to adapt to Keras/TF Keras, this code is modified from: https://github.com/CyberZHG/keras-adaptive-softmax

    """Turns dense vectors into probabilities.
    # Arguments
        input_dim: int > 0. Dimension of input vectors.
        output_dim: int > 0. Number of output classes.
        embed_dim: int > 0. Dimension of the dense embedding.
        cutoffs: list of ints. Indices of splitting points.
        div_val: int >= 0. The scaling parameter of embedding.
        use_bias: Boolean. Whether to bias terms.
        force_projection: Boolean. Add projection even if output_dim equals to embed_dim.
        bind_embeddings: list of boolean. Whether to use the existed embeddings as mapping.
        bind_projections: list of boolean. Whether to use the existed projections as mapping.
    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    # References
        - [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf)
    """

    def __init__(self, input_dim, output_dim, embed_dim=None,
                 cutoffs=None, div_val=1, use_bias=True,
                 force_projection=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 bind_embeddings=False,
                 bind_projections=False,
                 **kwargs):
        super(AdaptiveSoftmax, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if embed_dim is None:
            self.embed_dim = input_dim
        self.cutoffs = cutoffs
        if cutoffs is not None:
            if self.cutoffs[0] != 0:
                self.cutoffs = [0] + self.cutoffs
            if self.cutoffs[-1] != output_dim:
                self.cutoffs.append(output_dim)
        self.div_val = div_val
        self.use_bias = use_bias
        self.force_projection = force_projection
        if force_projection is None:
            if div_val == 1:
                self.force_projection = False
            else:
                self.force_projection = True
        self.cluster_num = 0
        if self.cutoffs is not None:
            self.cluster_num = len(self.cutoffs) - 2

        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.bind_embeddings = bind_embeddings
        if not isinstance(bind_embeddings, list):
            self.bind_embeddings = [bind_embeddings] * (self.cluster_num + 1)
        self.bind_projections = bind_projections
        if not isinstance(bind_projections, list):
            self.bind_projections = [bind_projections] * (self.cluster_num + 1)

        self.embeddings, self.projections, self.biases = (None,) * 3
        self.kernel_cluster, self.bias_cluster = None, None

    def build(self, input_shape):
        if self.div_val == 1:
            if not self.bind_embeddings[0]:
                self.embeddings = self.add_weight(
                    shape=(self.output_dim, self.embed_dim),
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint,
                    name='embeddings',
                )
            if self.embed_dim != self.input_dim or self.force_projection:
                if not self.bind_projections[0]:
                    self.projections = self.add_weight(
                        shape=(self.embed_dim, self.input_dim),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        name='kernel',
                    )
            if self.use_bias:
                self.biases = self.add_weight(
                    shape=(self.output_dim,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name='bias',
                )
        else:
            self.kernel_cluster = self.add_weight(
                shape=(self.embed_dim, self.cluster_num),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel-cluster',
            )
            if self.use_bias:
                self.bias_cluster = self.add_weight(
                    shape=(self.cluster_num,),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='bias-cluster',
                )
            self.embeddings, self.projections = [], []
            if self.use_bias:
                self.biases = []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                if self.bind_embeddings[i]:
                    self.embeddings.append(None)
                else:
                    self.embeddings.append(self.add_weight(
                        shape=(self.cutoffs[i + 1] - self.cutoffs[i], embed_dim),
                        initializer=self.embeddings_initializer,
                        regularizer=self.embeddings_regularizer,
                        constraint=self.embeddings_constraint,
                        name='embeddings-{}'.format(i),
                    ))
                if self.bind_projections[i]:
                    self.projections.append(None)
                else:
                    if embed_dim != self.input_dim or self.force_projection:
                        self.projections.append(self.add_weight(
                            shape=(embed_dim, self.input_dim),
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint,
                            name='kernel-{}'.format(i),
                        ))
                    else:
                        self.projections.append(None)
                if self.use_bias:
                    self.biases.append(self.add_weight(
                        shape=(self.cutoffs[i + 1] - self.cutoffs[i],),
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        name='bias-{}'.format(i),
                    ))
        super(AdaptiveSoftmax, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return mask[0]
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.output_dim,)

    def call(self, inputs, **kwargs):
        embeddings = inputs[1:1 + (self.cluster_num + 1)]
        projections = inputs[1 + (self.cluster_num + 1):]
        inputs = inputs[0]
        if self.div_val == 1:
            if self.embed_dim != self.input_dim or self.force_projection:
                projection = self.projections
                if projection is None:
                    projection = projections[0]
                inputs = K.dot(inputs, K.transpose(projection))
            embedding = self.embeddings
            if embedding is None:
                embedding = embeddings[0]
            out = K.dot(inputs, K.transpose(embedding))
            if self.use_bias:
                out = K.bias_add(out, self.biases)
            out = keras.activations.softmax(out, axis=-1)
        else:
            cluster_probs = None
            outputs = []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                if embed_dim != self.input_dim or self.force_projection:
                    projection = self.projections[i]
                    if projection is None:
                        projection = projections[i]
                    cluster_input = K.dot(inputs, K.transpose(projection))
                else:
                    cluster_input = inputs
                embedding = self.embeddings[i]
                if embedding is None:
                    embedding = embeddings[i]
                cluster_output = K.dot(cluster_input, K.transpose(embedding))
                if self.use_bias:
                    cluster_output = K.bias_add(cluster_output, self.biases[i])
                if cluster_probs is None:
                    cluster_probs = K.dot(cluster_input, self.kernel_cluster)
                    if self.use_bias:
                        cluster_probs = K.bias_add(cluster_probs, self.bias_cluster)
                    cluster_output = K.concatenate([cluster_output, cluster_probs], axis=-1)
                    cluster_output = keras.activations.softmax(cluster_output, axis=-1)
                    cluster_probs = cluster_output[..., -self.cluster_num:]
                    cluster_output = cluster_output[..., :-self.cluster_num]
                else:
                    cluster_output = keras.activations.softmax(cluster_output, axis=-1)
                    cluster_output = cluster_output * K.expand_dims(cluster_probs[..., i - 1])
                outputs.append(cluster_output)
            out = K.concatenate(outputs, axis=-1)

        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'cutoffs': self.cutoffs,
            'div_val': self.div_val,
            'use_bias': self.use_bias,
            'force_projection': self.force_projection,
            'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bind_embeddings': self.bind_embeddings,
            'bind_projections': self.bind_projections,
        }
        base_config = super(AdaptiveSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_custom_objects():
        return {'AdaptiveSoftmax': AdaptiveSoftmax}


class WithSparseCategoricalCrossEntropy(L.Layer):
    def call(self, inputs):
        y_true, y_pred, mask = inputs
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        self.add_loss(loss, inputs=inputs)
        return y_pred

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    @staticmethod
    def get_custom_objects():
        return {'WithSparseCategoricalCrossEntropy': WithSparseCategoricalCrossEntropy}


class WithSparseCategoricalAccuracy(L.Layer):
    def call(self, inputs):
        y_true, y_pred, mask = inputs
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        self.add_metric(acc, aggregation='mean', name=self.name)
        return y_pred

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    @staticmethod
    def get_custom_objects():
        return {'WithSparseCategoricalAccuracy': WithSparseCategoricalAccuracy}


class WithPerplexity(L.Layer):
    def call(self, inputs):
        y_true, y_pred, mask = inputs
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        ppl = K.exp(loss)
        self.add_metric(ppl, aggregation='mean', name=self.name)
        return y_pred

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    @staticmethod
    def get_custom_objects():
        return {'WithPerplexity': WithPerplexity}


custom_objects = {
    'swish': swish
}
custom_objects.update(AdaptiveSoftmax.get_custom_objects())
custom_objects.update(AdaptiveEmbedding.get_custom_objects())
custom_objects.update(WithSparseCategoricalCrossEntropy.get_custom_objects())
custom_objects.update(WithSparseCategoricalAccuracy.get_custom_objects())
custom_objects.update(WithPerplexity.get_custom_objects())
keras.utils.get_custom_objects().update(custom_objects)
