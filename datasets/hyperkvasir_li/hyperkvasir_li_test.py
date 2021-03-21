"""hyperkvasir_li dataset."""

import tensorflow_datasets as tfds
from . import hyperkvasir_li


class HyperkvasirLiTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for hyperkvasir_li dataset."""
  DATASET_CLASS = hyperkvasir_li.HyperkvasirLi
  SKIP_CHECKSUMS = True
  SPLITS = {
      'train': 9,  # Number of fake train example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()