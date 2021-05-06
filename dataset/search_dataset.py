from dataclasses import dataclass
from typing import List

import tensorflow as tf

from dataset.dataset_interface import CVDataset, DatasetSplits


@dataclass
class SearchDataset(CVDataset):
    cv_dataset: CVDataset

    '''
        Wrapper that modifies the train_dataset
        Make sure train_split and val_split are ~equal in size
        This requires special training step.
    '''

    def get_splits(self, batch_size: int, with_class_wight: bool = False) -> DatasetSplits:
        result = self.cv_dataset.get_splits(batch_size)

        result.train_split = tf.data.Dataset.zip((result.train_split, result.val_split))
        result.train_split_size = min(result.train_split_size, result.val_split_size)
        return result

    def number_of_classes(self):
        return self.cv_dataset.number_of_classes()

    def list_of_classes(self) -> List[str]:
        return self.cv_dataset.list_of_classes()

    def get_name(self) -> str:
        return self.cv_dataset.get_name()
