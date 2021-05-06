# https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/
# https://www.tensorflow.org/tutorials/load_data/images

from dataclasses import dataclass
from typing import List, Dict, Optional

import tensorflow as tf
from abc import ABC, abstractmethod


@dataclass
class DatasetSplits:
    train_split: tf.data.Dataset
    train_split_size: int
    train_class_weight: Optional[Dict[int, float]]
    val_split: tf.data.Dataset
    val_split_size: int
    test_split: tf.data.Dataset = None
    test_split_size: int = None


class CVDataset(ABC):
    """
    A Computer Vision Dataset
    """

    @abstractmethod
    def get_splits(self, batch_size: int, with_class_weight: bool = False) -> DatasetSplits:
        """ Return a DatasetSplits object """
        pass


    def number_of_classes(self):
        return len(self.list_of_classes())

    @abstractmethod
    def list_of_classes(self) -> List[str]:
        """Returns the classes as a list of strings"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of this dataset"""
        pass
