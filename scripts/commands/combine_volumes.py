import os.path
from os import PathLike
from typing import List, Union
import nibabel as nib


def combine_volumes(img_files: List[Union[str, PathLike]]):
    volumes = []
    for i in img_files:
        assert os.path.isfile(i)
        volumes.append(nib.load(i))
