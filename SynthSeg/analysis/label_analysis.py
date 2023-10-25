import os
from typing import List, Optional
import nibabel as nib
import nibabel.processing as proc
import numpy as np
from os import listdir
from os.path import isfile, join
import re
from dataclasses import dataclass, field
import logging
from simple_parsing import ArgumentParser
import sys


def setup_logger():
    our_logger = logging.getLogger("Analyse Label and Scan")
    our_logger.setLevel(logging.DEBUG)
    log_stdout_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter("%(levelname)s: %(message)s")
    log_stdout_handler.setFormatter(log_formatter)
    our_logger.addHandler(log_stdout_handler)
    return our_logger


# Global instance
logger = setup_logger()


def clip_and_rescale_nifti(nifti_file: str,
                           out_file: str,
                           min_clip: Optional[float] = None,
                           max_clip: Optional[float] = None,
                           min_out: float = 0.0,
                           max_out: float = 1.0):
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


def estimate_contrast_distribution(
        scan_file: str,
        label_file: str,
        percent_deviation: float = 5.0) -> dict:
    """
    Create contrast entries for given scan and label files where the label image should be a segmentation
    of the scan image. The idea behind this function is that you want to calculate the contrast statistics
    for each region specified by labels in `generation_labels`.
    The function will return a range for the mean and standard deviation of each region where the size of the range is
    determined by `percent_deviation`. These values are later used for generating training data for SynthSeg where
    the distribution of gray-values within a region is determined by randomly choosing a mean value from the range of
    mean values for the region and a randomly chosen standard deviation of each region.

    Args:
        scan_file (str): The file path of the scan.
        label_file (str): The file path of the label.
        percent_deviation (float, optional): The percentage deviation. Default is 5.0.

    Returns:
        dict: With the entries "output_labels", "generation_classes", "prior_means", and "prior_stds". All are input
        arguments for SynthSeg's brain generator.
    """

    import SynthSeg.analysis.freesurfer_tools as fsl_tools

    info = analyze_scan_and_label(scan_file, label_file)
    info = equalize_region_stats(info)
    assert set(info.keys()) == {"left_regions", "neutral_regions", "right_regions", "labels"}
    # labels = info["labels"]

    # Generate the `generation_labels` automatically by combining neutral labels, left labels, and right labels
    # in that order.
    generation_labels = [i.segmentation_class for name in
                         ["neutral_regions", "left_regions", "right_regions"] for i in info[name]]
    len_neutral_labels = len(info["neutral_regions"])
    len_left_labels = len(info["left_regions"])
    generation_classes_neutral = list(range(len_neutral_labels))
    generation_classes_left = list(range(len_neutral_labels, len_neutral_labels + len_left_labels))
    generation_classes = generation_classes_neutral + generation_classes_left + generation_classes_left
    min_prior_means = []
    max_prior_means = []
    min_prior_stds = []
    max_prior_stds = []

    def append_values(t: fsl_tools.TissueType):
        mean = t.mean
        std = t.std_dev
        min_prior_means.append(max(0.0, mean * (1.0 - percent_deviation / 100.0)))
        max_prior_means.append(min(1.0, mean * (1.0 + percent_deviation / 100.0)))
        min_prior_stds.append(max(0.0, std * (1.0 - percent_deviation / 100.0) * 0.001))
        max_prior_stds.append(min(1.0, std * (1.0 + percent_deviation / 100.0) * 0.001))

    for label_index in range(len_neutral_labels):
        current_type = info["neutral_regions"][label_index]
        append_values(current_type)

    for label_index in range(len_left_labels):
        current_type = info["left_regions"][label_index]
        append_values(current_type)

    # Convert numpy integers back to normal integers
    generation_labels = [num.item() for num in generation_labels]
    min_prior_means = [num.item() for num in min_prior_means]
    max_prior_means = [num.item() for num in max_prior_means]
    min_prior_stds = [num.item() for num in min_prior_stds]
    max_prior_stds = [num.item() for num in max_prior_stds]

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
    import SynthSeg.analysis.freesurfer_tools as fsl_tools

    label_img = nib.load(label_file)
    label_data = label_img.get_fdata()
    scan_img = nib.load(scan_file)
    scan_data = scan_img.get_fdata()
    resampled_labels = proc.resample_from_to(label_img, scan_img, order=0)
    # noinspection PyUnresolvedReferences
    resampled_labels_data = resampled_labels.get_fdata()
    labels = np.unique(label_data.flatten()).astype(np.int32)
    result = list(map(
        lambda label: fsl_tools.generate_tissue_types_from_sample(scan_data, resampled_labels_data, label), labels
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


def equalize_region_stats(regions_dict: dict) -> dict:
    """
    Equalizes the mean and standard deviation between corresponding left and right regions.
    The input should be the dictionary that is returned by `analyseLabelScanPair`.
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


@dataclass
class Options:
    label_file: str = None
    """The segmentation image with regions and labels according to FSL."""

    scan_files: List[str] = field(default_factory=list)
    """A list of n specific images with different contrast that are analysed.
     The label_file must be a segmentation of these images."""

    clip_min: List[float] = field(default_factory=list)
    """A list of n minimum clip values for each provided scan file."""

    clip_max: List[float] = field(default_factory=list)
    """A list of n maximum clip values for each provided scan file."""

    output_dir: str = None
    """Output directory for the analysis result and the rescaled scan files."""

    verbose: bool = False
    """Print debugging messages"""


if __name__ == '__main__':
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

    if isinstance(options.label_file, str) and os.path.isfile(options.label_file):
        logger.debug(f"Segmentation image: '{options.label_file}'")
    else:
        logger.error(f"Segmentation image does not exist: '{options.label_file}'")
        exit(1)

    if isinstance(options.scan_files, list):
        for image_num, f in enumerate(options.scan_files):
            if isinstance(f, str) and os.path.isfile(f):
                logger.debug(f"Using contrast image {image_num + 1}: '{f}'")
            else:
                logger.error(f"Contrast image does not exist: '{f}'")
                exit(1)

    num_contrasts = len(options.scan_files)
    clip_min_values = [None for i in range(num_contrasts)]
    clip_max_values = [None for i in range(num_contrasts)]

    # Check if clip values match the number of scans.
    if len(options.clip_min) == num_contrasts:
        clip_min_values = options.clip_min
    elif len(options.clip_min) == 0:
        logger.info("No min clip values provided. No clipping of values in the contrast images will happen.")
    else:
        logger.error("Either provide no min clip values or exactly as many as there are scan files.")
        exit(1)

    if len(options.clip_max) == num_contrasts:
        clip_max_values = options.clip_max
    elif len(options.clip_max) == 0:
        logger.info("No max clip values provided. No clipping of values in the contrast images will happen.")
    else:
        logger.error("Either provide no max clip values or exactly as many as there are scan files.")
        exit(1)

    # Window and rescale all scans and save them to the output directory.
    rescaled_contrast_images = []
    for image_num, f in enumerate(options.scan_files):
        file_name = os.path.basename(f)
        logger.info(f"Rescaling image {file_name}")
        output_file_path = os.path.join(options.output_dir, file_name)
        clip_and_rescale_nifti(f, output_file_path,
                               min_clip=clip_min_values[image_num], max_clip=clip_max_values[image_num])
        rescaled_contrast_images.append(output_file_path)

    # Analyze each scan with the label image and collect the results.
    statistics = []
    for f in rescaled_contrast_images:
        statistics.append(estimate_contrast_distribution(f, options.label_file, percent_deviation=5.0))

    import SynthSeg.brain_generator_options as gen

    gen_opts = gen.GeneratorOptions()
    gen_opts.n_channels = len(statistics)
    gen_opts.use_specific_stats_for_channel = True
    gen_opts.generation_labels = statistics[0]["generation_labels"]
    gen_opts.generation_classes = statistics[0]["generation_classes"]
    gen_opts.n_neutral_labels = statistics[0]["n_neutral_labels"]
    gen_opts.output_labels = statistics[0]["output_labels"]
    gen_opts.prior_means = statistics[0]["prior_means"]
    gen_opts.prior_stds = statistics[0]["prior_stds"]

    if len(statistics) > 1:
        for stat in statistics[1:]:
            gen_opts.prior_means = gen_opts.prior_means + stat["prior_means"]
            gen_opts.prior_stds = gen_opts.prior_stds + stat["prior_stds"]

    # Now we have the following problem: While we can perfectly determine all the labels that were used in the provided
    # segmentation, SynthSeg uses many more labels and needs definitions for them in order create synthetic images.
    # Right now, I don't have a good solution for that. What I will try out is to set unknown labels to background
    # so that they will have random background contrast and will not be used for the training.
    default_synth_seg_classes = [0, 14, 15, 16, 24, 72, 85, 502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 530, 2, 3,
                                 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 25, 26, 28, 30, 136, 137, 41, 42, 43, 44, 46, 47,
                                 49, 50, 51, 52, 53, 54, 57, 58, 60, 62, 163, 164]

    new_output_labels = []
    new_generation_classes = []
    for label in default_synth_seg_classes:
        if label in gen_opts.generation_labels:
            idx = gen_opts.generation_labels.index(label)
            new_output_labels.append(label)
            new_generation_classes.append(gen_opts.generation_classes[idx])
        else:
            new_output_labels.append(0)
            new_generation_classes.append(0)

    gen_opts.generation_labels = default_synth_seg_classes
    gen_opts.output_labels = new_output_labels
    gen_opts.generation_classes = new_generation_classes
    gen_opts.n_neutral_labels = 18

    # Write out the analysis result as a template brain generator config.
    gen_opts.save_yaml(os.path.join(options.output_dir, "generator.yml"), default_flow_style=None)
