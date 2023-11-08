from typing import Optional
import numpy as np
import nibabel as nib

from SynthSeg.logging_utils import get_logger
from freesurfer_tools import TissueType

logger = get_logger()


def clip_and_rescale_nifti(nifti_file: str,
                           out_file: str,
                           min_clip: Optional[float] = None,
                           max_clip: Optional[float] = None,
                           min_out: float = 0.0,
                           max_out: float = 255.0):
    """
    Windows and rescales the values in a NIfTI image to a specified range.
    This function helps if you want to prepare an image that contains outliers in, e.g., noisy regions.

    Args:
        nifti_file: The path to an existing NIfTI file to be rescaled.
        out_file: The path to save the rescaled NIfTI file. Directory needs to exist.
        min_clip: The minimum value to clip the data in the NIfTI file before rescaling.
        max_clip: The maximum value to clip the data in the NIfTI file before rescaling.
        min_out: The minimum value for rescaling the data. Default is 0.0.
        max_out: The maximum value for rescaling the data. Default is 255.0.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    clipped_data = np.clip(data, min_clip, max_clip)
    rescaled_data = min_out + ((clipped_data - min_clip) * (max_out - min_out)) / (max_clip - min_clip)
    nib.save(nib.Nifti1Image(rescaled_data, img.affine, img.header), out_file)


def calculate_winsorized_statistics(data: np.ndarray, percentile: float) -> tuple[float, float]:

    from scipy.stats.mstats import winsorize
    cleaned = data
    if 0.0 < percentile < 0.5:
        logger.info(f"Winsorize region data with {percentile}")
        cleaned = winsorize(data, limits=[percentile, percentile])
    else:
        logger.info("Percentile value must be positive and < 0.5. Skipping this step")
    return np.mean(cleaned), np.std(cleaned)


def generate_tissue_types_from_sample(scan_data: np.ndarray, segmentation_data: np.ndarray, label: int) -> TissueType:
    """
    Takes an existing segmentation for a scan and calculates statistical values for the segmentation class of the given
    `label`.
    The segmentation is expected to have class-labels according to FreeSurfer's brain segmentations.

    Args:
        scan_data (np.ndarray): The original brain-scan or an MPM map like PD, T1, etc.
        segmentation_data (np.ndarray): The segmentation of `scan_data`. Must have the same shape as `scan_data`.
        label: The tissue class to calculate the statistics for.

    Returns:
        TissueType: Statistics of the region with additional FreeSurfer metadata.
    """
    scan_data = scan_data
    seg_data = segmentation_data
    mask = seg_data == label
    data = scan_data[mask]

    from SynthSeg.analysis.freesurfer_tools import FSL_LUT, TissueType

    if label not in FSL_LUT.keys():
        print(f"Label number {label} not found in FSL lookup table. Using background for it!")
        label = 0
    lut_entry = FSL_LUT[label]
    win_mean, win_stddev = calculate_winsorized_statistics(data, 0.1)
    return TissueType(
        lut_entry, label,
        mean=np.mean(data),
        std_dev=np.std(data),
        perc_50=np.percentile(data, 50),
        perc_10=np.percentile(data, 10),
        perc_90=np.percentile(data, 90),
        win_mean=win_mean,
        win_stddev=win_stddev
    )


def equalize_region_stats(regions_dict: dict) -> dict:
    """
    Equalizes the mean and standard deviation between corresponding left and right regions.
    The input should be the dictionary returned by `analyseLabelScanPair`.
    The reasoning here is that when creating training data, we don't want to have different random
    distributions for corresponding left/right regions.

    Args:
        regions_dict (dict): A dictionary with keys: ['left_regions', 'neutral_regions', 'right_regions'].

    Returns:
        dict: The updated regions dictionary with equalized mean and standard deviation values.
    """
    assert set(regions_dict.keys()) == {"left_regions", "neutral_regions", "right_regions", "labels"}
    left_regions = regions_dict["left_regions"]
    right_regions = regions_dict["right_regions"]
    assert len(left_regions) == len(right_regions)
    for i in range(len(left_regions)):
        left_name = left_regions[i].label.name
        right_name = left_name.replace("Left-", "Right-").replace("ctx-lh", "ctx-rh")
        assert right_regions[i].label.name == right_name
        new_mean = 0.5 * (left_regions[i].mean + right_regions[i].mean)
        new_std_dev = 0.5 * (left_regions[i].std_dev + right_regions[i].std_dev)
        left_regions[i].mean = new_mean
        right_regions[i].mean = new_mean
        left_regions[i].std_dev = new_std_dev
        right_regions[i].std_dev = new_std_dev
    return regions_dict
