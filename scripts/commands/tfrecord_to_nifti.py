import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simple_parsing import ArgumentParser

from SynthSeg.logging_utils import get_logger
from SynthSeg.option_utils import get_absolute_path

logger = get_logger(os.path.basename(__file__))


@dataclass
class Options:
    """
    Converts a TF record to Nifti files
    """

    input_file: Optional[str] = None
    """
    Path to the training data directory that should contain the directories "images" and "labels" with matching
    Nifti filenames.
    """

    output_directory: Optional[str] = None


def main():
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(Options, "general")
    args = parser.parse_args()
    arguments: Options = args.general

    if not arguments.input_file:
        logger.error("Please provide a valid .tfrecord file")
        exit(-1)

    tfrecord_file = get_absolute_path(arguments.input_file)
    if isinstance(tfrecord_file, str) and os.path.isfile(tfrecord_file) and os.access(tfrecord_file, os.R_OK):
        logger.info(f"Seems I can read {tfrecord_file}")
    else:
        logger.error(f"Could not read {tfrecord_file}")
        exit(-1)

    output_path = Path(arguments.output_directory or os.path.dirname(tfrecord_file))
    output_path.mkdir(parents=True, exist_ok=True)
    if not os.access(output_path, os.W_OK):
        logger.error(f"Output directory {output_path} is not writable")
        exit(-1)

    image_output_path = output_path / "images"
    image_output_path.mkdir(parents=True, exist_ok=True)

    labels_output_path = output_path / "labels"
    labels_output_path.mkdir(parents=True, exist_ok=True)

    from SynthSeg.brain_generator import read_tfrecords
    from lab2im import utils

    dataset = read_tfrecords([tfrecord_file])
    for i, (image, labels) in enumerate(dataset.as_numpy_iterator()):
        file_name = os.path.basename(tfrecord_file).rsplit('.', 1)[0] + f"_{i}"
        logger.info(f"Exporting image {i}: {file_name}")
        utils.save_volume(
            np.squeeze(image, axis=0),
            None,
            None,
            str(image_output_path / f"{file_name}.nii.gz"),
        )
        utils.save_volume(
            np.squeeze(labels, axis=0),
            None,
            None,
            str(labels_output_path / f"{file_name}.nii.gz"),
        )


if __name__ == '__main__':
    main()
