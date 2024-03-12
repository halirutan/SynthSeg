import pathlib
from typing import List
import pytest

import tensorflow as tf
from . import TestData
import nibabel as nib
import numpy as np
import SynthSeg.brain_generator as bg
import SynthSeg.brain_generator_options as bg_opts
import timeit
from SynthSeg.logging_utils import get_logger

logger = get_logger("BrainGeneratorTest")


def get_generator_config(name: str) -> bg_opts.GeneratorOptions:
    generator_config_file = TestData.get_test_data_dir() / "generator_configs" / name
    generator_options = (bg_opts.GeneratorOptions.load(generator_config_file)
                         .with_absolute_paths(str(generator_config_file))
                         .convert_lists_to_numpy())
    return generator_options


def test_brain_generator():
    tf.keras.utils.set_random_seed(43)
    brain_generator = bg.create_brain_generator(get_generator_config("t1w.yml"))
    im, lab = brain_generator.generate_brain()
    if TestData.debug_nifti_output:
        # TODO: This won't work because im and lab are not TF tensors.
        img_data = tf.squeeze(im)
        nib.save(
            nib.Nifti1Image(img_data.numpy(), np.eye(4)),
            TestData.get_test_output_dir() / "generated_brain.nii",
        )
    assert (
            im.shape == lab.shape
    ), "Shape of the label image and the generated MRI are not the same"


def test_batchsize_contrast_behavior():
    test_batch_sizes = [1, 2, 4]
    first_images = []

    change_batch_size_in_generator = True

    if change_batch_size_in_generator:
        config = get_generator_config("t1w_no_augment.yml")
        brain_generator = bg.create_brain_generator(config)

    for batchsize in test_batch_sizes:
        tf.keras.utils.set_random_seed(43)
        if change_batch_size_in_generator:
            brain_generator.batchsize = batchsize
        else:
            config = get_generator_config("t1w_no_augment.yml")
            config.batchsize = batchsize
            brain_generator = bg.create_brain_generator(config)
        im, _ = brain_generator.generate_brain()
        if batchsize == 1:
            first_images.append(im)
        else:
            first_images.append(im[0])

    if TestData.debug_nifti_output:
        for i, img in enumerate(first_images):
            nib.save(
                nib.Nifti1Image(img, np.eye(4)),
                TestData.get_test_output_dir() / f"batched_comparison_{i}.nii",
            )

    for comparison_image in first_images[1:]:
        np.testing.assert_array_equal(first_images[0], comparison_image)


def test_tfrecords():
    tf.keras.utils.set_random_seed(43)
    generator_config = get_generator_config("t1w.yml")
    brain_generator = bg.create_brain_generator(generator_config)

    image, labels = brain_generator.generate_brain()

    tf.keras.utils.set_random_seed(43)
    tfrecord = TestData.get_test_output_dir("t1w.yml") / "test.tfrecord"
    brain_generator.generate_tfrecord(tfrecord)
    image2, labels2 = brain_generator.tfrecord_to_brain(tfrecord)

    for i in range(image.shape[0]):
        nib.save(
            nib.Nifti1Image(image[i], np.eye(4)),
            TestData.get_test_output_dir() / f"brain_{i}.nii",
        )

        nib.save(
            nib.Nifti1Image(image2[i], np.eye(4)),
            TestData.get_test_output_dir() / f"brain_tfrecords_{i}.nii",
        )

    np.testing.assert_array_equal(image, image2)
    np.testing.assert_array_equal(labels, labels2)


def test_tfrecords_compression(tmp_path):
    # TODO: Use the sample config instead of instantiating the brain generator over and over again with manual specs.
    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0], target_res=8, batchsize=2)

    tf.keras.utils.set_random_seed(43)
    tfrecord = TestData.get_test_output_dir() / "test.tfrecord"
    brain_generator.generate_tfrecord(tfrecord)

    tf.keras.utils.set_random_seed(43)
    tfrecord2 = TestData.get_test_output_dir() / "test2.tfrecord"
    brain_generator.generate_tfrecord(tfrecord2, compression_type="GZIP")

    assert tfrecord.stat().st_size / tfrecord2.stat().st_size > 2


def test_read_tfrecords(tmp_path):
    def measure_iteration(ds):
        def func():
            for _ in ds:
                pass

        return func

    label_map_files = TestData.get_label_maps()
    # TODO: Use the sample config instead of instantiating the brain generator over and over again with manual specs.
    brain_generator = bg.BrainGenerator(label_map_files[0], target_res=8, batchsize=2)

    tf.keras.utils.set_random_seed(43)
    for i in range(10):
        tfrecord = tmp_path / f"test{i}.tfrecord"
        brain_generator.generate_tfrecord(tfrecord, compression_type="GZIP")
    files = list(tmp_path.glob("*.tfrecord"))

    dataset = bg.read_tfrecords(files, compression_type="GZIP")
    time1 = timeit.timeit(measure_iteration(dataset), number=10)

    dataset = bg.read_tfrecords(
        files,
        num_parallel_reads=2,
        compression_type="GZIP",
    )
    time2 = timeit.timeit(measure_iteration(dataset), number=10)

    print(time1, time2)
    assert time1 > time2


def create_cube_test_data(
        means: List[float],
        stds: List[float],
        output_dir: pathlib.Path) -> dict:
    """
    Creates a label image that consists of cube-rings with different labels.
    Additionally, it provides a brain generator config supposed to create a gray level image from the segmentation and
    the provided means and std-deviations.
    Lastly, this function will also provide a reference image for the one the brain generator will provide.

    The goal is to have data to compare the outcome of the brain generator and to ensure that gray levels are what we
    expect.
    Args:
        means: List of means that are used to create random gray levels from a Gaussian distribution.
        stds: List of standard deviations with the same length as the means.
        output_dir: The directory where the output will be saved

    Returns:
        A dict containing the label image, the created gray level image and the brain generator config.
    """
    size = 10
    assert len(means) == len(stds), f"Length of means and stds must be the same: {len(means)} != {len(stds)}"
    assert len(means) > 0, "There should be at least 1 mean and std"
    data = np.zeros((size, size, size), dtype=np.int32)
    for level in range(1, len(means)):
        data = np.pad(data, ((size, size), (size, size), (size, size)), constant_values=level)
    label_file = output_dir / "test_cube_label.nii"
    nifti_label = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti_label, label_file)
    gray_levels = np.zeros_like(data, dtype=np.float32)
    masks = []
    for level in range(len(means)):
        masks.append(data == level)
    for i, m in enumerate(masks):
        gray_levels[m] = np.random.normal(means[i], stds[i], np.sum(m))
    img_file = output_dir / "test_cube.nii"
    nifti_img = nib.Nifti1Image(gray_levels, np.eye(4))
    nib.save(nifti_img, img_file)

    # Export scaled version
    img_file_scaled = output_dir / "test_cube_scaled.nii"
    nifti_img_scaled = nib.Nifti1Image(gray_levels/np.max(gray_levels), np.eye(4))
    nib.save(nifti_img_scaled, img_file_scaled)

    # TODO: Put it into a config like above
    from SynthSeg.brain_generator_options import GeneratorOptions
    config = GeneratorOptions()
    config.labels_dir = label_file
    config.bias_field_std = 0.0
    config.bias_scale = 0.0
    config.flipping = False
    config.n_neutral_labels = 0
    config.data_res = None
    config.target_res = 1.0
    config.randomise_res = False
    config.max_res_iso = 1.0
    config.max_res_iso = 1.0
    config.mix_prior_and_random = False
    config.n_channels = 1
    config.use_specific_stats_for_channel = True
    config.prior_distributions = "uniform"
    config.rotation_bounds = 0
    config.scaling_bounds = 0.0
    config.shearing_bounds = 0.0
    config.translation_bounds = 0.0
    config.nonlin_std = 0.0
    config.nonlin_scale = 0.0

    label_list = list(range(len(means)))
    config.generation_labels = label_list
    config.output_labels = label_list
    config.generation_classes = label_list
    config.prior_means = [means, means]
    config.prior_stds = [stds, stds]

    config_file = output_dir / "generator.yml"
    config.save_yaml(config_file, default_flow_style=None)
    return {
        "label": label_file,
        "image": img_file,
        "image_scaled": img_file_scaled,
        "config": config_file
    }


def check_region_distribution(
        label_data: np.ndarray,
        img_data: np.ndarray,
        label_to_check: int,
        expected_mean: float,
        expected_std: float,
        eps: float):
    masked_data = img_data[label_data == label_to_check]
    real_mean = np.mean(masked_data)
    real_std = np.std(masked_data)
    mean_error = np.abs(real_mean - expected_mean)
    std_error = np.abs(real_std - expected_std)
    try:
        assert mean_error <= eps
    except AssertionError:
        logger.warning(f"Mean for label {label_to_check} deviates by {mean_error} which is larger than {eps}")
        return False
    try:
        assert std_error <= eps
    except AssertionError:
        logger.warning(f"Standard deviation for label {label_to_check} deviates by {std_error} which is larger than {eps}")
        return False
    return True


def test_brain_generator_value():
    output_dir = TestData.get_test_output_dir("generator_test")
    means = [50.0, 100.0, 150.0, 200.0, 250.0]
    stds = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    info = create_cube_test_data(means, stds, output_dir)
    from SynthSeg.brain_generator_options import GeneratorOptions
    from SynthSeg.brain_generator import create_brain_generator
    generator_config = GeneratorOptions.load(info["config"]).convert_lists_to_numpy()
    generator = create_brain_generator(generator_config)
    from scripts.slurm.generation import _generate_image_label_pair
    _generate_image_label_pair(generator, output_dir, "generated")
    true_img = nib.load(info["image"]).get_fdata()
    gen_img_file = output_dir / "images" / f"generated.nii.gz"
    gen_img = nib.load(gen_img_file).get_fdata()
    true_label = nib.load(info["label"]).get_fdata()
    gen_label_file = output_dir / "labels" / f"generated.nii.gz"
    gen_label = nib.load(gen_label_file).get_fdata()

    assert np.array_equal(true_label, gen_label), \
        (f"Label image of our ground truth ({info['label']}) and the generated training label ({gen_label_file} "
         f"are not the same")
    for i in range(len(means)):
        try:
            assert check_region_distribution(true_label, true_img, i, means[i], stds[i], 0.01)
        except AssertionError:
            logger.warning(f"The ground truth label {i} doesn't have the expected distribution")
            pytest.fail()

        try:
            if i == 0:
                logger.info("Skipping check for label 0 on generated data because SynthSeg introduces artificial"
                            "augmentation for the background region that is random and can't be tested.")
            else:
                assert check_region_distribution(true_label, gen_img, i, means[i], stds[i], 0.01)
        except AssertionError:
            logger.warning(f"The brain generator label {i} doesn't have the expected distribution")
            pytest.fail()
