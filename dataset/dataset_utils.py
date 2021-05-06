from dataclasses import dataclass
from typing import Tuple, List

import tensorflow as tf

import dataset.transforms as transforms

AUTOTUNE = tf.data.experimental.AUTOTUNE


@dataclass
class DSConfigurator:
    batch_size: int
    drop_remainder: bool = False
    one_hot: bool = False
    number_of_classes: int = None
    repeat: bool = True
    augment: bool = False
    batch_transforms: List = None
    drop_path_prob: float = None
    cache: bool = True
    cache_file_name: str = None
    shuffle: bool = True
    seed: int = 42
    std: bool = True
    resize_spec: Tuple[int, int] = None

    def apply_config(self, not_configured_ds: tf.data.Dataset, buffer_size: int) -> tf.data.Dataset:

        result = not_configured_ds

        result = result.apply(transforms.Rescale())

        if self.one_hot and self.number_of_classes:
            result = result.apply(transforms.OneHotLabel(self.number_of_classes))

        if self.resize_spec:
            result = result.apply(transforms.Resize(self.resize_spec))

        if self.cache:
            result = result.apply(transforms.Cache(self.cache_file_name))

        if self.std:
            result = result.apply(transforms.PerImageStd())

        if self.repeat:
            result = result.apply(transforms.Repeat())

        if self.shuffle:
            result = result.apply(transforms.Shuffle(buffer_size, self.seed, False))

        if self.augment:
            result = result.apply(transforms.AugmentStateless(self.seed))

        result = result.apply(transforms.Batch(self.batch_size, self.drop_remainder))

        if self.drop_path_prob:
            result = result.apply(transforms.ZipConstantToInput(tf.constant([[[[self.drop_path_prob]]]])))

        result = result.apply(transforms.Prefetch())
        return result
