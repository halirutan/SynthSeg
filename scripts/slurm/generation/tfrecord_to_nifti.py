from dataclasses import dataclass
from pathlib import Path
import tarfile

import numpy as np
import simple_parsing
from tqdm.auto import tqdm


NUM_IMAGE_LABEL_PAIRS = 8


@dataclass
class Options:
    """Script to convert TFRecords to NIfTIs."""

    # Can be either a path to a single TFRecord file or a folder containing TFRecord files.
    input: Path
    # Path to the brain generator config file, used to generate the TFRecords.
    config: Path
    # Path to the output folder.
    output: Path
    # Skip the first N files
    skip: int = 0
    # Tar the images and labels into one tar file. In each TFRecord file there are 8 image/label pairs.
    tar: bool = False
    # Delete the TFRecord file after converting it to NifTIs?
    delete: bool = False


def to_niftis(
    tfrecord_path: Path,
    brain_generator_config: Path,
    output_path: Path,
    tar: bool = False,
    delete: bool = False,
):
    """Convert a TFRecord file into NIfTI files.

    Each image in the TFRecord file is appended by `_{i}_image.nii.gz` and the labels by `_{i}_labels.nii.gz`.

    Args:
        tfrecord_path: Path to the TFRecord file.
        brain_generator_config: Path to the brain generator config file.
        output_path: Path to the output folder.
        tar: Tar the NIfTIs into one tar file.
    """
    from ext.lab2im.layers import ConvertLabels
    from ext.lab2im.utils import save_volume
    from SynthSeg.brain_generator import create_brain_generator, read_tfrecords
    from SynthSeg.brain_generator_options import GeneratorOptions

    output_path.mkdir(parents=True, exist_ok=True)

    brain_generator = create_brain_generator(
        GeneratorOptions.load(brain_generator_config)
    )
    brain_generator.batchsize = 1
    iterator = read_tfrecords([str(tfrecord_path)]).as_numpy_iterator()

    for i, (image, labels) in tqdm(enumerate(iterator), total=NUM_IMAGE_LABEL_PAIRS):
        output_labels = np.unique(brain_generator.output_labels)
        labels = ConvertLabels(
            np.arange(len(output_labels)), dest_values=output_labels
        )(labels).numpy()
        image, labels = brain_generator._put_in_native_space(
            image[None, ...], labels[None]
        )

        save_volume(
            image,
            brain_generator.aff,
            brain_generator.header,
            str(output_path / f"{tfrecord_path.stem}_{i}_image.nii.gz"),
        )
        save_volume(
            labels,
            brain_generator.aff,
            brain_generator.header,
            str(output_path / f"{tfrecord_path.stem}_{i}_labels.nii.gz"),
        )

    if tar:
        with tarfile.open(f"{output_path / tfrecord_path.stem}.tar", "w") as tarf:
            for file in sorted(
                list(output_path.glob(f"{tfrecord_path.stem}_*.nii.gz"))
            ):
                tarf.add(file, arcname=file.name)
                file.unlink()

    if delete:
        tfrecord_path.unlink()


if __name__ == "__main__":
    opts = simple_parsing.parse(Options)
    if opts.input.is_file():
        to_niftis(opts.input, opts.config, opts.output, opts.tar, opts.delete)
    else:
        file_list = sorted(list(opts.input.glob("*.tfrecord")))
        for i, file in tqdm(enumerate(file_list)):
            if i < opts.skip:
                continue
            to_niftis(file, opts.config, opts.output, opts.tar, opts.delete)
