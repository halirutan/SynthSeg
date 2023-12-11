from dataclasses import dataclass, field
import os.path
from os import PathLike
from pathlib import Path
from typing import List, Union, Optional
import nibabel as nib
import numpy as np

from SynthSeg.analysis.contrast_analysis import clip_and_rescale_intensity
from SynthSeg.logging_utils import get_logger

logger = get_logger("combine_volumes.py")


def check_file_is_writable(file_path: Union[str, PathLike]) -> None:
    if not isinstance(file_path, PathLike):
        file_path = Path(file_path)

    if file_path.suffix not in ['.nii', '.nii.gz']:
        raise ValueError(f"File extension {file_path.suffix} is not allowed. Only '.nii' and '.nii.gz' are allowed.")

    if not os.access(file_path.parent, os.W_OK):
        raise PermissionError(f"Directory {file_path.parent} is not writable.")


@dataclass
class RescaleOptions:
    """Options for rescaling and clipping volumes."""

    rescale_range: Optional[List[float]] = None
    """
    Rescale range for the gray-values.
    If not specified then[0.0, 255.0] is used. If set to [], no rescaling is performed.
    """

    clip_min: Optional[List[float]] = None
    """A list of n minimum clip values for each provided contrast image."""

    clip_max: Optional[List[float]] = None
    """A list of n maximum clip values for each provided contrast image."""


def combine_volumes(img_files: List[Union[str, PathLike]],
                    output_file: Union[str, PathLike],
                    clip_min: List[float] = None,
                    clip_max: List[float] = None,
                    rescale_range: List[float] = None) -> None:
    n = len(img_files)
    check_file_is_writable(output_file)

    if n == 0:
        raise RuntimeError("No image files provided. That doesn't make sense, does it?")

    assert isinstance(clip_min, list) or clip_min is None, "clip_min must either be a list of floats or None."
    assert isinstance(clip_max, list) or clip_max is None, "clip_max must either be a list of floats or None."
    assert rescale_range is None or (isinstance(rescale_range, list) and len(rescale_range) == 2), \
        "rescale_range must either be a list of two floats or None."

    skip_rescaling = rescale_range is None and clip_min is None and clip_max is None

    if clip_min is None:
        clip_min = [None] * n
    if clip_max is None:
        clip_max = [None] * n
    if rescale_range is None:
        rescale_range = [None, None]

    if len(clip_min) != len(img_files) or len(clip_max) != len(img_files):
        raise ValueError("Clip min and max need to be lists of the same length as the number of images.")

    volumes = []
    for i in img_files:
        assert os.path.isfile(i), f"Image file {i} does not exist."
        volumes.append(nib.load(i))

    logger.info("Loading numpy arrays from NIFTI files")
    data = [v.get_fdata() for v in volumes]

    if not all(d.shape == data[0].shape for d in data):
        raise ValueError("Not all data arrays have the same shape")

    if skip_rescaling:
        logger.info("Skipping rescaling because no sensible options were provided")
    else:
        for i in range(len(data)):
            logger.info(f"Clipping and rescaling volume {i}")
            data[i] = clip_and_rescale_intensity(
                data[i],
                min_clip=clip_min[i],
                max_clip=clip_max[i],
                min_out=rescale_range[0],
                max_out=rescale_range[1]
            )

    logger.info(f"Exporting multi-channel volume to {output_file}")
    # Put the channels in the last dimension
    data = np.transpose(np.array(data), (1, 2, 3, 0))
    nib.save(nib.Nifti1Image(data, affine=volumes[0].affine, header=volumes[0].header, dtype=float), output_file)
