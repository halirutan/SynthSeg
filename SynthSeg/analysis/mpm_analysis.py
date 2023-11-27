import os
from simple_parsing import ArgumentParser

from SynthSeg.logging_utils import get_logger
from SynthSeg.analysis.analysis_types import Options, StatisticsOptions

logger = get_logger()


def main():
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(Options, "general")
    # noinspection PyTypeChecker
    parser.add_arguments(StatisticsOptions, "statistics")
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
    for i, f in enumerate(rescaled_contrast_images):
        settings: StatisticsOptions = args.statistics
        logger.info(f"Calculate '{settings.method}' statistic for contrast {i + 1} with parameter {settings.parameter} "
                    f"and range bracket '{settings.range_brackets}'.")
        statistics.append(
            estimate_contrast_distribution(
                f,
                options.label_file,
                settings=args.statistics,
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
    # Right now, I don't have a good solution for that.
    # I will provide two ways how to determine statistics for unknown regions.
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

    # Right now, we're using the default labels that SynthSeg needs.
    # However, since these labels are defined as "all labels available in the training segmentation maps", this code
    # needs to be adapted when we switch to a different set of training label maps.
    from SynthSeg.analysis.label_analysis import default_synth_seg_classes

    for label in default_synth_seg_classes:
        if label in gen_opts.generation_labels:
            idx = gen_opts.generation_labels.index(label)
            new_output_labels.append(label)
            new_generation_classes.append(gen_opts.generation_classes[idx])
        else:
            # What happens if SynthSeg requires statistics about a label that we don't find in the given segmentation?
            # Two ways: (1) regard them as the background and give them the same label id 0 and statistics of the
            # measured background region.
            # (2) Give each region a new gray value statistic that has a wide range so that the network essentially sees
            # it as random noise but different from the real background.
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
    output_file_generator = os.path.join(options.output_dir, "_generator.yml")
    logger.info(f"Writing generator config to {output_file_generator}")
    gen_opts.save_yaml(output_file_generator, default_flow_style=None)


if __name__ == '__main__':
    main()
