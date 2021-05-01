import tensorflow as tf
from absl import logging
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Flatten, Conv2D, MaxPool2D,
                                     GlobalAveragePooling2D, Softmax)

from modules.genotypes import PRIMITIVES, Genotype
from modules.operations import (OPS, FactorizedReduce, ReLUConvBN,
                                BatchNormalization, kernel_init)


def channel_shuffle(x, groups):
    _, h, w, num_channels = x.shape

    assert num_channels % groups == 0
    channels_per_group = num_channels // groups

    x = tf.reshape(x, [-1, h, w, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, num_channels])

    return x


class MixedOP(tf.keras.layers.Layer):
    """Mixed OP"""

    def __init__(self, ch, strides, name='MixedOP', **kwargs):
        super(MixedOP, self).__init__(name=name, **kwargs)

        self.ch = ch
        self.strides = strides
        self._ops = []
        self.mp = MaxPool2D(2, strides=2, padding='valid')

        for primitive in PRIMITIVES:
            op = OPS[primitive](self.ch // 2, self.strides, False)

            if 'pool' in primitive:
                op = Sequential([op, BatchNormalization(affine=False)])

            self._ops.append(op)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "ch": self.ch,
            "strides": self.strides
        })
        return config

    def call(self, inputs, **kwargs):
        x, weights = inputs
        # channel proportion k = 4
        x_1 = x[:, :, :, :x.shape[3] // 2]
        x_2 = x[:, :, :, x.shape[3] // 2:]

        # TODO why zip(tf.split()) ?
        x_1 = tf.add_n([w * op(x_1) for w, op in
                        zip(tf.split(weights, len(PRIMITIVES)), self._ops)])

        # reduction cell needs pooling before concat
        if x_1.shape[2] == x.shape[2]:
            # TODO why axis=3 ?
            ans = tf.concat([x_1, x_2], axis=3)
        else:
            ans = tf.concat([x_1, self.mp(x_2)], axis=3)

        return channel_shuffle(ans, 2)


class Cell(tf.keras.layers.Layer):
    """Cell Layer"""

    def __init__(self, steps, multiplier, ch, reduction, reduction_prev,
                 name='Cell', **kwargs):
        super(Cell, self).__init__(name=name, **kwargs)

        self.steps = steps
        self.multiplier = multiplier
        self.ch = ch
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        if self.reduction_prev:
            self.preprocess0 = FactorizedReduce(self.ch, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(self.ch, k=1, s=1, affine=False)
        self.preprocess1 = ReLUConvBN(self.ch, k=1, s=1, affine=False)

        self._ops = []
        for i in range(self.steps):
            for j in range(2 + i):
                strides = 2 if self.reduction and j < 2 else 1
                op = MixedOP(self.ch, strides=strides)
                self._ops.append(op)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "steps": self.steps,
            "multiplier": self.multiplier,
            "ch": self.ch,
            "reduction": self.reduction,
            "reduction_prev": self.reduction_prev
        })
        return config

    def call(self, inputs, **kwargs):
        s0, s1, weights, edge_weights = inputs
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _ in range(self.steps):
            s = 0
            for j, h in enumerate(states):
                branch = self._ops[offset + j]((h, weights[offset + j]))
                s += edge_weights[offset + j] * branch
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self.multiplier:], axis=-1)


class SplitSoftmax(tf.keras.layers.Layer):
    """Split Softmax Layer"""

    def __init__(self, size_splits, name='SplitSoftmax', **kwargs):
        super(SplitSoftmax, self).__init__(name=name, **kwargs)
        self.size_splits = size_splits
        self.soft_max_func = Softmax(axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "size_splits": self.size_splits
        })
        return config

    def call(self, value, **kwargs):
        return tf.concat(
            [self.soft_max_func(t) for t in tf.split(value, self.size_splits)],
            axis=0)


class SearchODSNetArch(object):
    """Search Other Dataset Network Architecture"""

    def __init__(self, cfg, steps=4, multiplier=4, stem_multiplier=3,
                 name='SearchModel'):

        self.input_size = cfg['input_size']
        self.init_channels = cfg['init_channels']
        self.cell_layers = cfg['layers']
        self.classes = cfg['num_classes']
        self.steps = steps
        self.multiplier = multiplier
        self.stem_multiplier = stem_multiplier
        self.name = name

        self.arch_parameters = self._initialize_alphas()
        self.model = self._build_model()

    def _initialize_alphas(self):
        k = sum(range(2, 2 + self.steps))
        num_ops = len(PRIMITIVES)
        w_init = tf.random_normal_initializer()
        self.alphas_normal = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k, num_ops], dtype='float32'),
            trainable=True, name='alphas_normal')
        self.alphas_reduce = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k, num_ops], dtype='float32'),
            trainable=True, name='alphas_reduce')
        self.betas_normal = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k], dtype='float32'),
            trainable=True, name='betas_normal')
        self.betas_reduce = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k], dtype='float32'),
            trainable=True, name='betas_reduce')

        return [self.alphas_normal, self.alphas_reduce, self.betas_normal,
                self.betas_reduce]

    def _build_model(self):
        """Model"""
        logging.info(f"buliding {self.name}...")

        # define model
        inputs = Input([self.input_size, self.input_size, 3], name='input_image')
        alphas_normal = Input([None], name='alphas_normal')
        alphas_reduce = Input([None], name='alphas_reduce')
        betas_normal = Input([], name='betas_normal')
        betas_reduce = Input([], name='betas_reduce')

        alphas_reduce_weights = Softmax(
            name='AlphasReduceSoftmax')(alphas_reduce)
        alphas_normal_weights = Softmax(
            name='AlphasNormalSoftmax')(alphas_normal)
        betas_reduce_weights = SplitSoftmax(
            range(2, 2 + self.steps), name='BetasReduceSoftmax')(betas_reduce)
        betas_normal_weights = SplitSoftmax(
            range(2, 2 + self.steps), name='BetasNormalSoftmax')(betas_normal)

        ch_curr = self.stem_multiplier * self.init_channels
        s0 = Sequential(
            [
                Conv2D(filters=ch_curr // 2, kernel_size=3, strides=2, padding='same',
                       kernel_initializer=kernel_init(), use_bias=False),
                BatchNormalization(affine=True),
                Conv2D(filters=ch_curr // 2, kernel_size=3, strides=2, padding='same',
                       kernel_initializer=kernel_init(), use_bias=False),
                BatchNormalization(affine=True),
            ],
            name='stem0')(inputs)

        s1 = Sequential([
            Conv2D(filters=ch_curr, kernel_size=3, strides=2, padding='same',
                   kernel_initializer=kernel_init(), use_bias=False),
            BatchNormalization(affine=True)], name='stem1')(s0)

        ch_curr = self.init_channels
        reduction_prev = True
        for layer_index in range(self.cell_layers):
            if layer_index in [self.cell_layers // 3, 2 * self.cell_layers // 3]:
                ch_curr *= 2
                reduction = True
                weights = alphas_reduce_weights
                edge_weights = betas_reduce_weights
            else:
                reduction = False
                weights = alphas_normal_weights
                edge_weights = betas_normal_weights

            cell = Cell(self.steps, self.multiplier, ch_curr, reduction,
                        reduction_prev, name=f'Cell_{layer_index}')
            s0, s1 = s1, cell((s0, s1, weights, edge_weights))

            reduction_prev = reduction

        fea = GlobalAveragePooling2D()(s1)

        logits = Dense(self.classes, kernel_initializer=kernel_init())(Flatten()(fea))


        return Model(
            (inputs, alphas_normal, alphas_reduce, betas_normal, betas_reduce),
            logits, name=self.name)

    def get_genotype(self):
        """get genotype"""

        def _parse(weights, edge_weights):
            n = 2
            start = 0
            gene = []
            for i in range(self.steps):
                end = start + n
                w = weights[start:end].copy()
                ew = edge_weights[start:end].copy()

                # fused weights
                for j in range(n):
                    w[j, :] = w[j, :] * ew[j]

                # pick the top 2 edges (k = 2).
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(w[x][k] for k in range(len(w[x]))
                                       if k != PRIMITIVES.index('none'))
                )[:2]

                # pick the top best op, and append into genotype.
                for j in edges:
                    k_best = None
                    for k in range(len(w[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or w[j][k] > w[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))

                start = end
                n += 1

            return gene

        gene_reduce = _parse(
            Softmax()(self.alphas_reduce).numpy(),
            SplitSoftmax(range(2, 2 + self.steps))(self.betas_reduce).numpy())
        gene_normal = _parse(
            Softmax()(self.alphas_normal).numpy(),
            SplitSoftmax(range(2, 2 + self.steps))(self.betas_normal).numpy())

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat,
                            reduce=gene_reduce, reduce_concat=concat)

        return genotype
