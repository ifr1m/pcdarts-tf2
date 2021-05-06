from typing import List, Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.engine import data_adapter

import dataset.transforms as transforms
from dataset.dataset_interface import CVDataset, DatasetSplits


class HyperkvasirLI(CVDataset):

    def __init__(self, seed: int, drop_path_prob: float = None,
                 fold_mapping=None):
        if fold_mapping is None:
            fold_mapping = {"train": "split_0", "val": "split_1"}

        self.seed = seed
        self.name = "hyperkvasir_li"
        self.train_split = fold_mapping["train"]
        self.val_split = fold_mapping["val"]
        self.test_split = fold_mapping["val"]  # use the same split as for validation (not ideal..)
        # self.train_split = "split_0[0:10%]"
        # self.val_split = "split_1[0:10%]"
        # self.test_split = self.val_split
        self.target_size = (224, 224)
        self.train_ds, self.train_split_size, self.classes = load_tfds(self.name, seed, self.train_split)
        self.val_ds, self.val_split_size, _ = load_tfds(self.name, seed, self.val_split)
        self.test_ds, self.test_split_size, _ = load_tfds(self.name, seed, self.test_split)
        self.drop_path_prob = drop_path_prob

    def get_splits(self, batch_size: int, with_class_weight: bool = False) -> DatasetSplits:
        train = self.train_ds.apply(self._train_ds_transformer(batch_size, self.train_split_size))
        val = self.val_ds.apply(self._val_ds_transformer(batch_size, self.val_split_size))
        test = self.test_ds.apply(self._test_ds_transformer(batch_size, self.val_split_size))

        train_class_weight = None
        if with_class_weight:
            train_class_weight = get_class_weight(self.train_ds, self.number_of_classes())

        return DatasetSplits(train_split=train, train_split_size=self.train_split_size,
                             train_class_weight=train_class_weight,
                             val_split=val, val_split_size=self.val_split_size,
                             test_split=test, test_split_size=self.test_split_size)

    def _train_ds_transformer(self, batch_size, shuffle_buffer_size) -> transforms.DSTransform:
        return transforms.Compose([
            transforms.Rescale(),
            transforms.OneHotLabel(self.number_of_classes()),
            transforms.Resize(self.target_size),
            transforms.Cache(),
            transforms.PerImageStd(),
            transforms.Repeat(),
            transforms.Shuffle(shuffle_buffer_size, self.seed, False),
            transforms.AugmentStateless(self.seed),
            transforms.Batch(batch_size, False),
            transforms.ZipConstantToInput(tf.constant([[[[self.drop_path_prob]]]])) if self.drop_path_prob else None,
            transforms.Prefetch()
        ])

    def _val_ds_transformer(self, batch_size, shuffle_buffer_size) -> transforms.DSTransform:
        return transforms.Compose([
            transforms.Rescale(),
            transforms.OneHotLabel(self.number_of_classes()),
            transforms.Resize(self.target_size),
            transforms.Cache(),
            transforms.PerImageStd(),
            transforms.Repeat(),
            transforms.Shuffle(shuffle_buffer_size, self.seed, False),
            transforms.Batch(batch_size, False),
            transforms.ZipConstantToInput(tf.constant([[[[self.drop_path_prob]]]])) if self.drop_path_prob else None,
            transforms.Prefetch()
        ])

    def _test_ds_transformer(self, batch_size, shuffle_buffer_size) -> transforms.DSTransform:
        return transforms.Compose([
            transforms.Rescale(),
            transforms.OneHotLabel(self.number_of_classes()),
            transforms.Resize(self.target_size),
            transforms.PerImageStd(),
            transforms.Batch(batch_size, False),
            transforms.Prefetch()
        ])

    def list_of_classes(self) -> List[str]:
        return self.classes

    def get_name(self) -> str:
        return self.name


class HyperkvasirLISearch(HyperkvasirLI):
    """Specially crafted to work with NAS DARTS networks."""

    def get_splits(self, batch_size: int, with_class_weight: bool = False) -> DatasetSplits:
        train_net = self.train_ds.apply(self._train_ds_transformer(batch_size, self.train_split_size))
        # custom add class weight to the train_net dataset, it wont work if sample_weight is also needed
        if with_class_weight:
            train_class_weight = get_class_weight(self.train_ds, self.number_of_classes())
            train_net = train_net.map(data_adapter._make_class_weight_map_fn(train_class_weight))

        train_arch = self.val_ds.apply(self._train_ds_transformer(batch_size, self.val_split_size))
        val = self.val_ds.apply(self._val_ds_transformer(batch_size, self.val_split_size))

        train_zipped_size = min(self.train_split_size, self.val_split_size)

        start_arch_training_ds = self._start_arch_training_ds(train_zipped_size // batch_size, 35)

        train_zipped = tf.data.Dataset.zip((train_net, train_arch, start_arch_training_ds))

        return DatasetSplits(train_split=train_zipped, train_split_size=train_zipped_size,
                             train_class_weight=None,
                             val_split=val, val_split_size=self.val_split_size)

    def _constant_ds(self, value):
        return tf.data.Dataset.from_tensor_slices([value]).repeat()

    def _start_arch_training_ds(self, steps_per_epoch, start_epoch):
        return self._constant_ds(0).take(steps_per_epoch * (start_epoch - 1)).concatenate(self._constant_ds(1))



def get_class_weight(ds: tf.data.Dataset, number_of_classes) -> Dict[int, float]:
    """ ds should not be batched and labels not encoded in one_hot vector"""
    # Wee need to iterate over the dataset, there is no other way currently. Some stats on tfds would help.
    counts = tf.reduce_sum(list(ds.apply(transforms.OneHotLabel(number_of_classes)).map(lambda i, l: l)),
                           axis=0).numpy()
    weights = np.clip(np.log(max(counts) / counts), 1, 50)
    return {i: weights[i] for i in range(len(weights))}


def load_tfds(dataset_name: str, shuffle_seed: int, split='train'):
    dataset_split, info = tfds.load(name=dataset_name, split=split, as_supervised=True, with_info=True,
                                    shuffle_files=True,
                                    read_config=tfds.ReadConfig(
                                        shuffle_seed=shuffle_seed,
                                        skip_prefetch=True,
                                    ))

    split_size = info.splits[split].num_examples
    classes = info.features["label"].names

    return dataset_split, split_size, classes
