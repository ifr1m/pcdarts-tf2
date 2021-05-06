from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DSTransform(ABC):

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return self.transform(dataset)

    @abstractmethod
    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        pass


@dataclass
class Resize(DSTransform):
    resize_spec: Tuple[int, int]

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(self.resize, num_parallel_calls=AUTOTUNE)

    def resize(self, image, label):
        # resize the image to the desired size.
        return tf.image.resize(image, [self.resize_spec[0], self.resize_spec[1]]), label


class PerImageStd(DSTransform):

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(self.std, num_parallel_calls=AUTOTUNE)

    def std(self, image, label):
        image = tf.image.per_image_standardization(image)
        return image, label


class Rescale(DSTransform):

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(self.rescale, num_parallel_calls=AUTOTUNE)

    def rescale(self, image, label):
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        return tf.image.convert_image_dtype(image, tf.float32), label


@dataclass
class OneHotLabel(DSTransform):
    number_of_classes: int

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(
            lambda image, label: (image, tf.one_hot(label, depth=self.number_of_classes, dtype=tf.int32)))


@dataclass
class Cache(DSTransform):
    cache_file_name: str = None

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self.cache_file_name is not None:
            return dataset.cache(self.cache_file_name)
        else:
            return dataset.cache()


class Repeat(DSTransform):
    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.repeat()


@dataclass
class Shuffle(DSTransform):
    buffer_size: int
    seed: int
    reshuffle_each_iteration: bool

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed, reshuffle_each_iteration=False)


@dataclass
class AugmentStateless(DSTransform):
    seed: int

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        seeds_ds = tf.data.experimental.RandomDataset(seed=self.seed).batch(2)
        dataset = tf.data.Dataset.zip((seeds_ds, dataset))
        return dataset.map(self.augment_data, num_parallel_calls=AUTOTUNE, deterministic=True)

    def augment_data(self, seeds, element):
        images, labels = element
        images = tf.image.stateless_random_flip_left_right(images, seeds)
        images = tf.image.stateless_random_flip_up_down(images, seeds)
        images = self.stateless_random_rot90(images, seeds)
        # image = tf.image.stateless_random_saturation(image, 1, 1.3)
        # image = tf.image.stateless_random_contrast(image, 1, 1.3)
        # image = tf.image.stateless_random_brightness(image, 0.2)
        return images, labels

    @staticmethod
    def stateless_random_rot90(images, seeds):
        # 0 (0), 1 (90), 2 (180), 3 (270)
        k = tf.random.stateless_uniform(shape=[], seed=seeds, minval=0, maxval=4, dtype=tf.int32)
        images = tf.image.rot90(images, k)
        return images


@dataclass
class Batch(DSTransform):
    batch_size: int
    drop_remainder: bool

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)


@dataclass
class ZipConstantToInput(DSTransform):
    constant: tf.Tensor

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda image, label: ((image, self.constant), label))


class Prefetch(DSTransform):
    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.prefetch(buffer_size=AUTOTUNE)

@dataclass
class Compose(DSTransform):
    transformers: List[DSTransform]

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        result = dataset
        for t in self.transformers:
            if t is not None:
                result = result.apply(t)
        return result
