import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, SeparableConv2D, MaxPool2D,
                                     AveragePooling2D, ReLU)

OPS = {'none': lambda f, s, affine: Zero(s),
       'avg_pool_3x3': lambda f, s, affine:
       AveragePooling2D(3, strides=s, padding='same'),
       'max_pool_3x3': lambda f, s, affine:
       MaxPool2D(3, strides=s, padding='same'),
       'skip_connect': lambda f, s, affine:
       Identity() if s == 1 else FactorizedReduce(f, affine=affine),
       'sep_conv_3x3': lambda f, s, affine:
       SepConv(f, 3, s, affine=affine),
       'sep_conv_5x5': lambda f, s, affine:
       SepConv(f, 5, s, affine=affine),
       'sep_conv_7x7': lambda f, s, affine:
       SepConv(f, 7, s, affine=affine),
       'dil_conv_3x3': lambda f, s, affine:
       DilConv(f, 3, s, 2, affine=affine),
       'dil_conv_5x5': lambda f, s, affine:
       DilConv(f, 5, s, 2, affine=affine),
       'conv_7x1_1x7': lambda f, s, affine:
       Sequential([ReLU(),
                   Conv2D(filters=f, kernel_size=(1, 7), strides=(1, s),
                          kernel_initializer=kernel_init(),
                          padding='same', use_bias=False),
                   Conv2D(filters=f, kernel_size=(7, 1), strides=(s, 1),
                          kernel_initializer=kernel_init(),
                          padding='same', use_bias=False),
                   BatchNormalization(affine=affine)])}


def kernel_init(seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal(seed)


def drop_path(x, drop_rate, name='drop_path'):
    """drop path from https://arxiv.org/abs/1605.07648"""
    random_values = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], name=name)
    mask = tf.cast(random_values > drop_rate, tf.float32) / (1 - drop_rate)
    x = mask * x

    return x


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, affine=True,
                 name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=affine,
            scale=affine, name=name, **kwargs)


class ReLUConvBN(tf.keras.layers.Layer):
    """ReLu + Conv + BN"""

    def __init__(self, ch_out, k, s, padding='valid', affine=True,
                 name='ReLUConvBN', **kwargs):
        super(ReLUConvBN, self).__init__(name=name, **kwargs)
        self.op = Sequential([
            ReLU(),
            Conv2D(filters=ch_out, kernel_size=k, strides=s,
                   kernel_initializer=kernel_init(),
                   padding=padding, use_bias=False),
            BatchNormalization(affine=affine)])

    def call(self, x, **kwargs):
        return self.op(x)


# TODO grouping ?

class DilConv(tf.keras.layers.Layer):
    """Dilated Conv"""

    def __init__(self, ch_out, k, s, d, affine=True, name='DilConv',
                 **kwargs):
        super(DilConv, self).__init__(name=name, **kwargs)

        self.op = Sequential([
            ReLU(),
            Conv2D(filters=ch_out, kernel_size=k, strides=1,
                   kernel_initializer=kernel_init(),
                   dilation_rate=d, padding='same', use_bias=False),
            Conv2D(filters=ch_out, kernel_size=1,
                   kernel_initializer=kernel_init(), padding='same', strides=s, use_bias=False),
            BatchNormalization(affine=affine)])

    def call(self, x, **kwargs):
        return self.op(x)


class SepConv(tf.keras.layers.Layer):
    """Separable Conv"""

    def __init__(self, ch_out, k, s, affine=True, name='SepConv',
                 **kwargs):
        super(SepConv, self).__init__(name=name, **kwargs)
        self.op = Sequential([
            ReLU(),
            SeparableConv2D(filters=ch_out, kernel_size=k, strides=s,
                            depthwise_initializer=kernel_init(),
                            pointwise_initializer=kernel_init(),
                            padding='same', use_bias=False),
            BatchNormalization(affine=affine),
            ReLU(),
            SeparableConv2D(filters=ch_out, kernel_size=k, strides=1,
                            depthwise_initializer=kernel_init(),
                            pointwise_initializer=kernel_init(),
                            padding='same', use_bias=False),
            BatchNormalization(affine=affine)])

    def call(self, x, **kwargs):
        return self.op(x)


class Identity(tf.keras.layers.Layer):
    """Identity"""

    def __init__(self, name='Identity', **kwargs):
        super(Identity, self).__init__(name=name, **kwargs)

    def call(self, x, **kwargs):
        return x


class Zero(tf.keras.layers.Layer):
    """Zero"""

    def __init__(self, strides, name='Zero', **kwargs):
        super(Zero, self).__init__(name=name, **kwargs)
        self.strides = strides

    def call(self, x, **kwargs):
        if self.strides == 1:
            return x * 0.
        return x[:, ::self.strides, ::self.strides, :] * 0


class FactorizedReduce(tf.keras.layers.Layer):
    """Factorized Reduce Layer"""

    def __init__(self, ch_out, affine=True, name='FactorizedReduce',
                 **kwargs):
        super(FactorizedReduce, self).__init__(name=name, **kwargs)
        assert ch_out % 2 == 0
        self.relu = ReLU()
        self.conv_1 = Conv2D(filters=ch_out // 2, kernel_size=1, strides=2,
                             kernel_initializer=kernel_init(),
                             padding='valid', use_bias=False)
        self.conv_2 = Conv2D(filters=ch_out // 2, kernel_size=1, strides=2,
                             kernel_initializer=kernel_init(),
                             padding='valid', use_bias=False)
        self.bn = BatchNormalization(affine=affine)

    def call(self, x, **kwargs):
        x = self.relu(x)
        out = tf.concat([self.conv_1(x), self.conv_2(x)], axis=-1)
        out = self.bn(out)

        return out
