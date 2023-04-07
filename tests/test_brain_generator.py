from . import TestData
import SynthSeg.brain_generator as bg
import numpy as np
from pathlib import Path
import os


def test_brain_generator(set_random_seeds):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    image_path = Path(__file__).parent / "brain_image_seed43.npy"
    label_path = Path(__file__).parent / "brain_label_seed43.npy"

    label_map_files = sorted(TestData.get_label_maps())

    brain_generator = bg.BrainGenerator(label_map_files[0], output_shape=16)
    image, label = brain_generator.generate_brain()

    # np.save(image_path, image)
    # np.save(label_path, label)

    target_image = np.load(str(image_path))
    target_label = np.load(str(label_path))

    assert np.allclose(image, target_image)
    assert np.allclose(label, target_label)
