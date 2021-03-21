"""hyperkvasir_li dataset."""

import os
from pathlib import Path

import tensorflow_datasets as tfds

_DESCRIPTION = """
Hyper-Kvasir: A Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy

This tfds contains only the labeled images (https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip)
"""

_CITATION = """
@misc{borgli2020, title={Hyper-Kvasir: A Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy}, url={osf.io/mkzcq}, DOI={10.31219/osf.io/mkzcq}, publisher={OSF Preprints}, author={Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, Sigrun L and Randel, Kristin R and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc T D and Johansen, Dag and Griwodz, Carsten and Stensland, H{\aa}kon K and Garcia-Ceja, Enrique and Schmidt, Peter T and Hammer, Hugo L and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas}, year={2019}, month={Dec}}
"""



class HyperkvasirLi(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for hyperkvasir_li dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    CLASSES = ["barretts",
               "barretts-short-segment",
               "bbps-0-1",
               "bbps-2-3",
               "cecum",
               "dyed-lifted-polyps",
               "dyed-resection-margins",
               "esophagitis-a",
               "esophagitis-b-d",
               "hemorrhoids",
               "ileum",
               "impacted-stool",
               "polyps",
               "pylorus",
               "retroflex-rectum",
               "retroflex-stomach",
               "ulcerative-colitis-grade-0-1",
               "ulcerative-colitis-grade-1",
               "ulcerative-colitis-grade-1-2",
               "ulcerative-colitis-grade-2",
               "ulcerative-colitis-grade-2-3",
               "ulcerative-colitis-grade-3",
               "z-line"]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(hyperkvasir_li): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3)),
                'label': tfds.features.ClassLabel(names=self.CLASSES),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage='https://datasets.simula.no/hyper-kvasir/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        extracted_dir = dl_manager.download_and_extract(
            'https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip')

        images_dir = Path(extracted_dir) / "labeled-images"

        return {
            'train': self._generate_examples(images_dir)
        }


    def get_label(self, file_path:Path):
        # convert the path to a list of path components
        parts = str(file_path).split(os.path.sep)
        # The second to last is the class-directory
        return parts[-2]


    def _generate_examples(self, dir_path: Path):
        img_types = ["*.jpg", "*.JPG", "*.png", "*.PNG"]
        """Yields examples."""
        for path_generators in [dir_path.rglob(e) for e in img_types]:
            for file_path in path_generators:
                label = self.get_label(file_path)
                yield file_path.name, {
                    'image': str(file_path),
                    'label': label,
                }
