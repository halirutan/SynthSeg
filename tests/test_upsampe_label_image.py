import numpy as np
import os
import nibabel as nib

import SynthSeg.analysis.resample_label_image as res
from tests import TestData


def get_first_map() -> nib.Nifti1Image:
    label_map = TestData.get_label_maps()[0]
    label_map_image: nib.Nifti1Image = nib.load(label_map)
    return label_map_image


def test_upsample():
    output_dir = TestData.get_test_output_dir("resample_label")
    resolution = 0.5
    label_image = get_first_map()
    output_image = res.resample_label_image(label_image, resolution)
    output_shape = np.array(output_image.shape)
    assert np.array_equal(output_shape, np.array(label_image.shape) * 2)
    # noinspection PyUnresolvedReferences
    zooms = np.array(output_image.header.get_zooms())
    assert np.all(zooms == resolution)
    nib.save(output_image, os.path.join(output_dir, "map01_resampled_500um.nii.gz"))


def test_upsample_cropped():
    output_dir = TestData.get_test_output_dir("resample_label")
    resolution = 0.5
    crop_size = 256
    label_image = get_first_map()
    output_image = res.resample_label_image_cropped(label_image, resolution, crop_size)
    output_shape = np.array(output_image.shape)
    assert np.all(output_shape == crop_size)
    # noinspection PyUnresolvedReferences
    zooms = np.array(output_image.header.get_zooms())
    assert np.all(zooms == resolution)
    nib.save(output_image, os.path.join(output_dir, "map01_resampled_500um_cropped_256.nii.gz"))
