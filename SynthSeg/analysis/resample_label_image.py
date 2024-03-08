import os
import numpy as np
import nibabel as nib
from typing import List, Union
from dataclasses import dataclass
from simple_parsing import ArgumentParser

from SynthSeg.logging_utils import get_logger

logger = get_logger("Resample")


def resample_label_image(nifti: nib.Nifti1Image, resolution_out: Union[float, List[float]]) -> nib.Nifti1Image:
    """
    Resample a label image to a new resolution.
    Note: This is intended for internal use only to upsample label images that are in a too low resolution.

    Args:
        nifti: The input label image represented as a nib.Nifti1Image object.
        resolution_out: The desired output resolution. It can be a single float value or a list of three float values
            representing the desired voxel size in mm in x, y, and z directions respectively.

    Returns:
        The resampled label image represented as a nib.Nifti1Image object.
    """
    if type(resolution_out) is float:
        resolution_out = np.array([resolution_out] * 3)
    else:
        resolution_out = np.array(resolution_out)
    data_resampled = do_resample(nifti, resolution_out)
    header = nifti.header
    # noinspection PyUnresolvedReferences
    header.set_zooms(resolution_out)
    return nib.Nifti1Image(data_resampled.astype(np.int32), nifti.affine, header)


def resample_label_image_cropped(
        nifti: nib.Nifti1Image,
        resolution_out: Union[float, List[float]],
        new_size: Union[int, List[int]]
) -> nib.Nifti1Image:

    """
    Resample a label image to a new resolution and crops it to a specified size.
    Note: This is intended for internal use only to upsample label images that are in a too low resolution.

    Args:
        nifti: The input label image represented as a nib.Nifti1Image object.
        resolution_out: The desired output resolution. It can be a single float value or a list of three float values
            representing the desired voxel size in mm in x, y, and z directions respectively.
        new_size: Desired size of the final image. It will be cropped from all sides, and the provided size must be
            smaller than the original image.

    Returns:
        The resampled label image represented as a nib.Nifti1Image object.
    """
    if type(resolution_out) is float:
        resolution_out = np.array([resolution_out] * 3)
    else:
        resolution_out = np.array(resolution_out)
    if type(new_size) is int:
        new_size = np.full(3, new_size)
    else:
        new_size = np.array(new_size)
    data_resampled = do_resample(nifti, resolution_out)
    size = (nz, ny, nx) = data_resampled.shape
    if len(new_size) != 3 or np.max(size - new_size) < 0:
        raise RuntimeError("Cropped size is expected to be a single positive integer or a list of 3 positive integers")
    (cz, cy, cx) = np.floor((size - new_size)/2).astype(np.intp)
    header = nifti.header
    # noinspection PyUnresolvedReferences
    header.set_zooms(resolution_out)
    return nib.Nifti1Image(data_resampled[cz:nz-cz, cy:ny-cy, cx:nx-cx].astype(np.int32), nifti.affine, header)


def do_resample(nifti: nib.Nifti1Image, resolution_out: np.ndarray) -> np.ndarray:
    header = nifti.header
    # noinspection PyUnresolvedReferences
    dim = header.get_data_shape()
    if len(dim) != 3:
        raise RuntimeError("Image data does not have 3 dimensions")
    # noinspection PyUnresolvedReferences
    resolution_in = header["pixdim"][1:4]
    step_size = resolution_out / resolution_in
    zs = np.arange(0, dim[0], step_size[0]).astype(np.intp)
    ys = np.arange(0, dim[1], step_size[1]).astype(np.intp)
    xs = np.arange(0, dim[2], step_size[2]).astype(np.intp)
    data: np.ndarray = nifti.get_fdata()
    data_resampled = data[np.ix_(zs, ys, xs)]
    return data_resampled


@dataclass
class Options:
    image_file: str
    """Input label image for rescaling."""

    output_dir: str
    """
    Output directory where to store the resampled image.
    """

    resolution: float
    """
    Resolution in mm for the resampled label image.
    """

    crop_size: int = None
    """If not None, it must be an int or a list of integers specifying the output size."""


def main():
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(Options, "general")
    args = parser.parse_args()
    options: Options = args.general

    if isinstance(options.output_dir, str) and os.path.isdir(options.output_dir):
        logger.debug(f"Using output directory: '{options.output_dir}'")
    else:
        logger.error(f"Output directory does not exist: '{options.output_dir}'")
        exit(1)

    if isinstance(options.image_file, str) and os.path.isfile(options.image_file):
        logger.debug(f"Using label image: '{options.image_file}'")
    else:
        logger.error(f"Provided image is not a regular file: '{options.image_file}'")
        exit(1)

    resolution = options.resolution
    crop_size = options.crop_size
    output_dir = options.output_dir

    image_file = options.image_file
    nifti = nib.load(image_file)
    if not isinstance(nifti, nib.Nifti1Image):
        logger.error(f"Image {image_file} is not a Nifti1 image")
        exit(1)

    if crop_size is not None:
        result = resample_label_image_cropped(nifti, resolution, crop_size)
    else:
        result = resample_label_image(nifti, resolution)

    if not isinstance(result, nib.Nifti1Image):
        logger.error("Unable to rescale image.")

    file_base = os.path.basename(image_file)
    output_file = os.path.join(output_dir, file_base)
    nib.save(result, output_file)


if __name__ == '__main__':
    main()
