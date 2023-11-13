from dataclasses import dataclass, field
from typing import List, Optional
import os
from simple_parsing import ArgumentParser

from SynthSeg.logging_utils import get_logger

logger = get_logger()


@dataclass
class Options:
    label_file: str = None
    """The segmentation image with regions and labels according to FSL."""

    contrast_files: List[str] = field(default_factory=list)
    """A list of n specific images with different contrast that are analysed.
     The label_file must be a segmentation of these images."""

    rescale_contrasts: bool = True

    clip_min: List[float] = field(default_factory=list)
    """A list of n minimum clip values for each provided contrast image."""

    clip_max: List[float] = field(default_factory=list)
    """A list of n maximum clip values for each provided contrast image."""

    output_dir: str = None
    """Output directory for the analysis result and the rescaled contrast image."""

    equalize_left_right_regions: bool = False
    """
    If true then equivalent regions in the left/right hemisphere will use the same gray-value distributions, 
    even if they are different in the measured contrast images. 
    """

    undefined_region_as_background: bool = True
    """
    If set to True, regions that where not found in the analysed segmentation will not be included in the training.
    If set to False, each undefined region will have a random gray-level distribution in the synthetic map defined
    by the undefined_region_stats option.
    However, these unknown labels will not be used for calculating the loss of the segmentation.
    """

    undefined_region_stats: List[float] = field(default_factory=lambda: [25.0, 225.0, 5.0, 25.0])
    """
    Labels that are required for training but could not be found in the analysed scans need gray-level distributions
    as well.
    Each of these regions will get a separate gray-level distribution, randomly drawn from the provided parameters
    which are in the form [min_mean, max_mean, min_stddev, max_stddev].
    """

    template_generator_config: str = ""
    """
    Provides a template generator configuration that is used. All values are preserved except the ones necessary for
    setting the segmentation labels and gray-value statistics.
    """

# TODO: Make statistics options work
@dataclass
class StatisticsOptions:
    """

    """

    method: str = "winsorized"
    """
    Defines how to calculate min/max for the Gaussian gray-level distribution from the data of each region.
    Each method calculates an estimate for the mean and standard deviation of the data.
    Possible values are "winsorized", "median", and "gaussian".
    Method "winsorized" first filters the data by removing outliers up until a specified percentile and then calculates
    the normal mean and standard deviation.
    Method "median" calculates the median of the data and uses the interquartile range with of a specific percentile
    to determine the standard deviation.
    Method "gaussian" calculates the mean and standard deviation on the unfiltered region data. This might lead to bad
    estimates if the data contains outliers or artefacts.
    """

    parameter: float = 0.9
    """
    Parameters for the different methods.
    For method "winsorized", a value specifying the percentages to cut on each side of the data,
    with respect to the number of unmasked data, as float between 0. and 1. Default 0.9.
    For method "median", the percentile to use for the IQR. Default 0.9
    For method "gaussian", this value is unused.
    """

    percentages: List[float] = field(default_factory=lambda: [0.95, 1.05, 0.5, 1.05])
    """
    """


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

    if isinstance(options.label_file, str) and os.path.isfile(options.label_file):
        logger.debug(f"Segmentation image: '{options.label_file}'")
    else:
        logger.error(f"Segmentation image does not exist: '{options.label_file}'")
        exit(1)

    if isinstance(options.contrast_files, list):
        for image_num, f in enumerate(options.contrast_files):
            if isinstance(f, str) and os.path.isfile(f):
                logger.debug(f"Using contrast image {image_num + 1}: '{f}'")
            else:
                logger.error(f"Contrast image does not exist: '{f}'")
                exit(1)

    num_contrasts = len(options.contrast_files)

    # Window and rescale all images and save them to the output directory.
    rescaled_contrast_images = []
    if options.rescale_contrasts:
        clip_min_values = [None for _ in range(num_contrasts)]
        clip_max_values = [None for _ in range(num_contrasts)]

        # Check if clip values match the number of images.
        if len(options.clip_min) == num_contrasts:
            clip_min_values = options.clip_min
        elif len(options.clip_min) == 0:
            logger.info("No min clip values provided. No clipping of values in the contrast images will happen.")
        else:
            logger.error("Either provide no min clip values or exactly as many as there are contrast files.")
            exit(1)

        if len(options.clip_max) == num_contrasts:
            clip_max_values = options.clip_max
        elif len(options.clip_max) == 0:
            logger.info("No max clip values provided. No clipping of values in the contrast images will happen.")
        else:
            logger.error("Either provide no max clip values or exactly as many as there are contrast images.")
            exit(1)

        from SynthSeg.analysis.contrast_analysis import clip_and_rescale_nifti

        for image_num, f in enumerate(options.contrast_files):
            file_name = os.path.basename(f)
            logger.info(f"Rescaling image {file_name}")
            output_file_path = os.path.join(options.output_dir, file_name)
            clip_and_rescale_nifti(f, output_file_path,
                                   min_clip=clip_min_values[image_num], max_clip=clip_max_values[image_num])
            rescaled_contrast_images.append(output_file_path)
    else:
        rescaled_contrast_images = options.contrast_files

    # Analyze each contrast with the label image and collect the results.
    from SynthSeg.analysis.label_analysis import estimate_contrast_distribution

    statistics = []
    for f in rescaled_contrast_images:
        logger.info(f"Calculate statistic for contrast in {os.path.basename(f)}")
        statistics.append(
            estimate_contrast_distribution(
                f,
                options.label_file,
                equalize_regions=options.equalize_left_right_regions
            )
        )

    import SynthSeg.brain_generator_options as gen

    gen_opts = gen.GeneratorOptions()
    if isinstance(options.template_generator_config, str) and os.path.isfile(options.template_generator_config):
        logger.info(f"Initialize generator config with values from {options.template_generator_config}")
        gen_opts = gen.GeneratorOptions.load(options.template_generator_config)
    else:
        logger.info("Initialize generator config with default values")

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

    # Now we have the following problem: While we can perfectly determine all the labels used in the provided
    # segmentation, SynthSeg uses many more labels and needs definitions for them to create synthetic images.
    # Right now, I don't have a good solution for that. What I will try out is to set unknown labels to background
    # so that they will have random background contrast and will not be used for the training.

    undefined_region = {}

    if not options.undefined_region_as_background:
        if isinstance(options.undefined_region_stats, list) and len(options.undefined_region_stats) == 4:
            logger.info("Using random values for unknown labels in synthetic images")
            undefined_region = {
                "min_mean": options.undefined_region_stats[0],
                "max_mean": options.undefined_region_stats[1],
                "min_stddev": options.undefined_region_stats[2],
                "max_stddev": options.undefined_region_stats[3]
            }
        else:
            # Someone did not set the option undefined_region_stats correctly. Giving up
            logger.error(f"When using option 'undefined_region_stats', it must be a list of 4 float values")
            exit(1)
    else:
        logger.info("Regarding unknown labels as background in the synthetic images")

    new_output_labels = []
    new_generation_classes = []
    new_idx = max(gen_opts.generation_classes) + 1

    from SynthSeg.analysis.label_analysis import default_synth_seg_classes

    for label in default_synth_seg_classes:
        if label in gen_opts.generation_labels:
            idx = gen_opts.generation_labels.index(label)
            new_output_labels.append(label)
            new_generation_classes.append(gen_opts.generation_classes[idx])
        else:
            if options.undefined_region_as_background:
                new_output_labels.append(0)
                new_generation_classes.append(0)
            else:
                new_output_labels.append(0)
                new_generation_classes.append(new_idx)
                new_idx += 1
                for i in range(gen_opts.n_channels):
                    # noinspection PyUnresolvedReferences
                    gen_opts.prior_means[2 * i].append(undefined_region["min_mean"])
                    # noinspection PyUnresolvedReferences
                    gen_opts.prior_means[2 * i + 1].append(undefined_region["max_mean"])
                    # noinspection PyUnresolvedReferences
                    gen_opts.prior_stds[2 * i].append(undefined_region["min_stddev"])
                    # noinspection PyUnresolvedReferences
                    gen_opts.prior_stds[2 * i + 1].append(undefined_region["max_stddev"])
    gen_opts.generation_labels = default_synth_seg_classes
    gen_opts.output_labels = new_output_labels
    gen_opts.generation_classes = new_generation_classes
    gen_opts.n_neutral_labels = 18

    # Write out the analysis result as a template brain generator config.
    gen_opts.save_yaml(os.path.join(options.output_dir, "generator.yml"), default_flow_style=None)


if __name__ == '__main__':
    main()
