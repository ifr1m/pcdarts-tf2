import copy
import itertools
from dataclasses import field
from pathlib import Path
from typing import Generator


def flat_map(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


def remove_none_types(list):
    return [x for x in list if x is not None]


def find_leaf_dirs(top: Path) -> Generator[Path, None, None]:
    for dirpath, dirnames, filenames in path_dir_walk(top):
        if not dirnames:
            yield dirpath


def path_dir_walk(top: Path, topdown=False, followlinks=False):
    names = list(top.iterdir())

    dirs = list((node for node in names if node.is_dir() is True))
    nondirs = list((node for node in names if node.is_dir() is False))

    if topdown:
        yield top, dirs, nondirs

    for name in dirs:
        if followlinks or name.is_symlink() is False:
            for x in path_dir_walk(name, topdown, followlinks):
                yield x

    if topdown is not True:
        yield top, dirs, nondirs


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))
