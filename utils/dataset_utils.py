from dataclasses import dataclass
from typing import Tuple, List

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


@dataclass
class DSConfigurator:
    batch_size: int
    drop_remainder: bool = False
    one_hot: bool = False
    number_of_classes: List[int] = None
    unpack_dict: bool = False
    repeat: bool = True
    augment: bool = False
    cache: bool = True
    cache_file_name: str = None
    shuffle: bool = True
    shuffle_seed: int = 42
    std: bool = True
    resize_spec: Tuple[int, int] = None

    def apply_config(self, not_configured_ds: tf.data.Dataset, buffer_size: int) -> tf.data.Dataset:

        result = not_configured_ds

        if self.unpack_dict:
            # unpack
            result = result.map(lambda features: (features['image'], features['label']))

        result = result.map(self.rescale, num_parallel_calls=AUTOTUNE)

        if self.one_hot and self.number_of_classes:
            result = result.map(lambda image, label: (image, tf.one_hot(label, depth=self.number_of_classes)))

        if self.resize_spec:
            result = result.map(self.resize, num_parallel_calls=AUTOTUNE)

        if self.cache:
            if self.cache_file_name is not None:
                result = result.cache(self.cache_file_name)
            else:
                result = result.cache()

        if self.augment:
            # it works on 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
            #       of shape `[height, width, channels]`.
            result = result.map(self.augment_data, num_parallel_calls=AUTOTUNE)

        if self.std:
            # it works on 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
            #       of shape `[height, width, channels]`.

            # result = result.map(lambda image, label: (self.std(image), label), num_parallel_calls=AUTOTUNE)
            result = result.map(self.stddd, num_parallel_calls=AUTOTUNE)

        if self.shuffle:
            result.shuffle(buffer_size=buffer_size, seed=self.shuffle_seed, reshuffle_each_iteration=True)

        if self.repeat:
            result = result.repeat()

        result = result.batch(self.batch_size, drop_remainder=self.drop_remainder)

        result = result.prefetch(buffer_size=AUTOTUNE)
        return result

    def augment_data(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_saturation(image, 1, 1.3)
        image = tf.image.random_contrast(image, 1, 1.3)
        image = tf.image.random_brightness(image, 0.2)
        return image, label

    def resize(self, image, label):
        # resize the image to the desired size.
        return tf.image.resize(image, [self.resize_spec[0], self.resize_spec[1]]), label

    def rescale(self, image, label):
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        return tf.image.convert_image_dtype(image, tf.float32), label

    def stddd(self, image, label):
        image = tf.image.per_image_standardization(image)
        return image, label
