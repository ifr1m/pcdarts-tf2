import unittest

from modules.dataset import load_hyperkvasir_dataset


class HyperkvasirLoadTest(unittest.TestCase):

    def testLoad(self):
        t_split = "train[0:10]"
        ds = load_hyperkvasir_dataset(batch_size=5, split=t_split, drop_remainder=False, repeat=False)
        elements = list(ds)
        assert len(elements) == 2
        assert elements[0][0].shape == (5, 224, 224, 3)
        assert elements[0][1].shape == (5, 23)
