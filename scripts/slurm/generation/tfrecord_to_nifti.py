from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from tqdm.auto import tqdm


@dataclass
class Options:
    """Script to convert tfrecords to niftis."""

    # Can be either a path to a single tfrecord file or a folder containing tfrecord files.
    input: Path
    # Path to the brain generator config file, used to generate the tfrecords.
    config: Path
    # Path to the output folder. In this folder we will create an 'images' and 'labels' folder.
    output: Path
    # Skip the first N files
    skip: int = 0


def to_niftis(tfrecord_path: Path, brain_generator_config: Path, output_path: Path):
    from ext.lab2im.layers import ConvertLabels
    from ext.lab2im.utils import save_volume
    from SynthSeg.brain_generator import create_brain_generator, read_tfrecords
    from SynthSeg.brain_generator_options import GeneratorOptions

    image_output_path = output_path / "images"
    image_output_path.mkdir(parents=True, exist_ok=True)
    labels_output_path = output_path / "labels"
    labels_output_path.mkdir(parents=True, exist_ok=True)

    brain_generator = create_brain_generator(
        GeneratorOptions.load(brain_generator_config)
    )
    brain_generator.batchsize = 1
    iterator = read_tfrecords([str(tfrecord_path)]).as_numpy_iterator()

    for i, (image, labels) in tqdm(enumerate(iterator)):
        output_labels = np.unique(brain_generator.output_labels)
        labels = ConvertLabels(
            np.arange(len(output_labels)), dest_values=output_labels
        )(labels).numpy()
        image, labels = brain_generator._put_in_native_space(
            image[None, ..., 0], labels[None]
        )

        save_volume(
            image,
            brain_generator.aff,
            brain_generator.header,
            str(image_output_path / f"{tfrecord_path.stem}_{i}.nii.gz"),
        )
        save_volume(
            labels,
            brain_generator.aff,
            brain_generator.header,
            str(labels_output_path / f"{tfrecord_path.stem}_{i}.nii.gz"),
        )


if __name__ == "__main__":
    opts = simple_parsing.parse(Options)
    if opts.input.is_file():
        to_niftis(opts.input, opts.config, opts.output)
    else:
        file_list = sorted(list(opts.input.glob("*.tfrecord")))
        for i, file in tqdm(enumerate(file_list)):
            if i < opts.skip:
                continue
            to_niftis(file, opts.config, opts.output)
