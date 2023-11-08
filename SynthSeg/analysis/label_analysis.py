import nibabel as nib
import nibabel.processing as proc
import numpy as np
from os import listdir
from os.path import isfile, join
import re

from SynthSeg.analysis.contrast_analysis import generate_tissue_types_from_sample
from SynthSeg.logging_utils import get_logger

logger = get_logger()

default_synth_seg_classes = [0, 14, 15, 16, 24, 72, 85, 502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 530, 2, 3,
                             4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 25, 26, 28, 30, 136, 137, 41, 42, 43, 44, 46, 47,
                             49, 50, 51, 52, 53, 54, 57, 58, 60, 62, 163, 164]


def estimate_contrast_distribution(
        contrast_file: str,
        label_file: str,
        equalize_regions: bool = True) -> dict:
    """
    Create contrast entries for given scan and label files where the label image should be a segmentation
    of the scan image. The idea behind this function is that you want to calculate the contrast statistics
    for each region specified by labels in `generation_labels`.
    The function will return a range for the mean and standard deviation of each region where the size of the range is
    determined by `percent_deviation`. These values are later used for generating training data for SynthSeg.
    The distribution of gray-values within a region is determined by randomly choosing a mean value from the range of
    mean values for the region and a randomly chosen standard deviation of each region.

    Args:
        contrast_file (str): The file path of the scan.
        label_file (str): The file path of the label.
        equalize_regions (bool): If true, the gray-level statistics of the left and right regions will be equalized.

    Returns:
        dict: With the entries "output_labels", "generation_classes", "prior_means", and "prior_stds". All are input
        arguments for SynthSeg's brain generator.
    """

    import SynthSeg.analysis.freesurfer_tools as fsl_tools
    from SynthSeg.analysis.contrast_analysis import equalize_region_stats

    info = analyze_scan_and_label(contrast_file, label_file)
    if equalize_regions:
        info = equalize_region_stats(info)
    assert set(info.keys()) == {"left_regions", "neutral_regions", "right_regions", "labels"}

    # Generate the `generation_labels` automatically by combining neutral labels, left labels, and right labels
    # in that order.
    generation_labels = [i.segmentation_class for name in
                         ["neutral_regions", "left_regions", "right_regions"] for i in info[name]]
    len_neutral_labels = len(info["neutral_regions"])
    len_left_labels = len(info["left_regions"])
    len_right_labels = len(info["left_regions"])
    assert len_right_labels == len_left_labels, "Different number of left and right regions."

    generation_classes_neutral = list(range(len_neutral_labels))

    if equalize_regions:
        generation_classes_left = list(range(len_neutral_labels, len_neutral_labels + len_left_labels))
        generation_classes = generation_classes_neutral + generation_classes_left + generation_classes_left
    else:
        generation_classes_left = list(
            range(len_neutral_labels, len_neutral_labels + len_left_labels + len_right_labels))
        generation_classes = generation_classes_neutral + generation_classes_left

    min_prior_means = []
    max_prior_means = []
    min_prior_stds = []
    max_prior_stds = []

    def calculate_and_append_statistic(t: fsl_tools.TissueType):
        """
            At this point, we need to decide from which values SynthSeg should randomly choose the gray-values for
            a specific region.
            The `TissueType` contains mean, std_dev and (10, 50, 90) percentiles of the data.
            SynthSeg wants to draw a mean and std_dev for each region in every synthetic image and needs lower and
            upper bounds for the values.
            I suggest using the 10 and 90 percentiles for the lower and upper bound of the "mean" because this is
            better, especially with the hot pixels in certain regions.
            For randomly choosing a standard deviation, I think the measured std_dev could be the upper bound, and
            we just decrease the lower bound to let the network also see not so noisy data.
        """
        min_prior_means.append(max(0.0, t.perc_10))
        max_prior_means.append(min(255.0, t.perc_90))
        min_prior_stds.append(1.0)
        max_prior_stds.append(min(255.0, t.std_dev))

    for label_index in range(len_neutral_labels):
        current_type = info["neutral_regions"][label_index]
        calculate_and_append_statistic(current_type)

    for label_index in range(len_left_labels):
        current_type = info["left_regions"][label_index]
        calculate_and_append_statistic(current_type)

    if not equalize_regions:
        for label_index in range(len_right_labels):
            current_type = info["right_regions"][label_index]
            calculate_and_append_statistic(current_type)

    # Convert numpy integers back to normal integers
    generation_labels = [int(num) for num in generation_labels]
    min_prior_means = [float(num) for num in min_prior_means]
    max_prior_means = [float(num) for num in max_prior_means]
    min_prior_stds = [float(num) for num in min_prior_stds]
    max_prior_stds = [float(num) for num in max_prior_stds]

    return {
        "generation_labels": generation_labels,
        "n_neutral_labels": len_neutral_labels,
        "output_labels": generation_labels,
        "generation_classes": generation_classes,
        "prior_means": [min_prior_means, max_prior_means],
        "prior_stds": [min_prior_stds, max_prior_stds]}


def analyze_scan_and_label(scan_file: str, label_file: str) -> dict:
    """
    Calculates region statistics and information for a given scan and segmentation image pair.
    Scan and label images are not required to have the same resolution, but they need to represent the same view.
    The label image is rescaled to the resolution of the scan image. After that, every available segmentation class
    in the label image is processed and FreeSurfer label information is added. It returns a dict containing sorted lists
    of "neutral_regions", "left_regions" and "right_regions".

    Args:
        scan_file: File path to the NIfTI scan file.
        label_file: File path to the NIfTI label file.

    Returns:
        A dictionary containing the analysis results of the label and scan pair. The dictionary has the following keys:
            - "neutral_regions": A sorted list of tissue types found in regions that are not specific to left or right.
            - "left_regions": A sorted list of tissue types found in regions specific to the left side.
            - "right_regions": A sorted list of tissue types found in regions specific to the right side.
    """
    import freesurfer_tools as fsl_tools

    label_img = nib.load(label_file)
    label_data = label_img.get_fdata()
    scan_img = nib.load(scan_file)
    scan_data = scan_img.get_fdata()
    resampled_labels = proc.resample_from_to(label_img, scan_img, order=0)
    # noinspection PyUnresolvedReferences
    resampled_labels_data = resampled_labels.get_fdata()
    labels = np.unique(label_data.flatten()).astype(np.int32)
    result = list(map(
        lambda label: generate_tissue_types_from_sample(scan_data, resampled_labels_data, label), labels
    ))

    left_regions = list(filter(lambda entry: re.match(fsl_tools.FSL_LEFT_LABEL_REGEX, entry.label.name), result))
    left_regions.sort(key=lambda entry: entry.segmentation_class)
    right_regions = list(filter(lambda entry: re.match(fsl_tools.FSL_RIGHT_LABEL_REGEX, entry.label.name), result))
    right_regions.sort(key=lambda entry: entry.segmentation_class)
    neutral_regions = [reg for reg in result if reg not in left_regions and reg not in right_regions]
    neutral_regions.sort(key=lambda entry: entry.segmentation_class)
    assert len(left_regions) == len(right_regions), \
        "There should be exactly as many left regions as there are right regions"
    return {
        "neutral_regions": neutral_regions,
        "left_regions": left_regions,
        "right_regions": right_regions,
        "labels": labels
    }


def list_available_labels_in_map(nifti_file: str) -> np.ndarray:
    """
    Reads in a segmentation NIfTI image and returns a list of all labels found in the image.

    Args:

        nifti_file (str): Path to the segmentation NIfTI image

    Returns:
        np.ndarray: Sorted numpy array of all found labels
    """
    label_map = nib.load(nifti_file)
    data = np.array(label_map.get_data(), dtype=np.int64)
    return np.unique(data)


def find_all_available_labels(directory: str) -> np.ndarray:
    """
    Does the same as `listAvailableLabelsInMap` but for a whole directory containing
    segmentation maps.

    Args:
        directory (str): Directory containing the segmentation maps

    Returns:
        np.ndarray: Sorted numpy array of all available labels
    """
    result = np.array([])
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    for f in files:
        labels = list_available_labels_in_map(f)
        result = np.append(result, labels)
    return np.unique(result)

