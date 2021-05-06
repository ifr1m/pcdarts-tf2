"""hyperkvasir_li dataset."""

import csv
import os
from csv import unix_dialect
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from pydblite import Base
from tensorflow_datasets.core.download import ExtractMethod


class UnixSemiColonDialect(unix_dialect):
    delimiter = ';'


csv.register_dialect("unix-semi-colon", UnixSemiColonDialect)

_DESCRIPTION = """
Hyper-Kvasir: A Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy

This tfds contains only the labeled images (https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip)
It contains the official 2 fold split (https://github.com/simula/hyper-kvasir)
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

    IMAGE_ARCHIVE_URL = "https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip"
    SPLITS_CSV_URL = "https://raw.githubusercontent.com/simula/hyper-kvasir/49c0d0f915b57ac0b6cb18b49776ffa61c5512e3/official_splits/2_fold_split.csv"

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
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
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://datasets.simula.no/hyper-kvasir/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        images_resource = tfds.download.Resource(url=self.IMAGE_ARCHIVE_URL, extract_method=ExtractMethod.ZIP)
        splits_csv_resource = tfds.download.Resource(url=self.SPLITS_CSV_URL)

        paths = dl_manager.download_and_extract(
            {self.IMAGE_ARCHIVE_URL: images_resource, self.SPLITS_CSV_URL: splits_csv_resource})

        print(paths)

        images_dir = Path(paths[self.IMAGE_ARCHIVE_URL]) / "labeled-images"
        print(images_dir)
        db = self.create_db_from_files(images_dir)
        spit_ids = self.split_ids(db, paths[self.SPLITS_CSV_URL])
        self._print_images_per_split(db, spit_ids)

        return {
            'split_0': self._generate_examples(db, spit_ids[0], images_dir),
            'split_1': self._generate_examples(db, spit_ids[1], images_dir)
        }

    def _print_images_per_split(self, db, spit_ids):
        for split, ids in spit_ids.items():
            print("-----------------------------------------------------------")
            print(f"split_{split} file names: {[db[id]['image_name'] for id in ids]}")

    def get_label(self, file_path: Path):
        # convert the path to a list of path components
        parts = str(file_path).split(os.path.sep)
        # The second to last is the class-directory
        return parts[-2]

    def _generate_examples(self, db, ids, images_dir: Path):
        for id in ids:
            r = db[id]
            label = r["label"]
            image_name = r["image_name"]
            file_path = r["path"]
            yield image_name, {
                'image': str(file_path),
                'label': label,
            }

    def split_ids(self, db, spits_csv_path):
        splits = {0: [], 1: []}

        with tf.io.gfile.GFile(spits_csv_path) as csv_f:
            reader = csv.DictReader(csv_f, dialect="unix-semi-colon")
            for row in reader:
                image_name = row["file-name"]
                split = int(row["split-index"])
                db_records = db(image_name=image_name)
                if len(db_records) == 1:
                    splits[split].append(db_records[0]["__id__"])
                else:
                    raise Exception(
                        f"Split image reference {row} not found(or found multiple times) on disk. Split csv file {spits_csv_path}. Probably the split csv file and the images archive are not in sync.")
        return splits

    def create_db_from_files(self, images_dir: Path) -> Base:
        db = Base('hyperkvasir-li-labels.pdl')
        db.create('image_name', 'label', 'path', mode="override")
        db.create_index('image_name')
        img_types = ["*.jpg", "*.JPG", "*.png", "*.PNG"]

        for path_generators in [images_dir.rglob(e) for e in img_types]:
            for file_path in path_generators:
                label = self.get_label(file_path)
                db.insert(image_name=file_path.name, label=label, path=str(file_path))
        db.commit()
        return db
