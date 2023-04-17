import pytest
import tensorflow as tf
import numpy as np
import random

from . import TestData
from SynthSeg.model_inputs import build_model_inputs


@pytest.fixture
def fixed_random_seed() -> None:
    """
        Initializes known random generators to make tests deterministic.
    """
    seed = 12345
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


@pytest.fixture(scope="session")
def model_inputs():
    """This fixture returns a list of inputs necessary for the label_to_image model.
    Returns:
        A List containing the input label map, as well as the means and stds defining the parameters of the GMM.
    """
    np.random.seed(43)

    labels_classes_path = TestData.synth_seg_path / "data" / "labels_classes_priors"
    generation_labels = np.load(labels_classes_path / "generation_labels.npy")
    generation_classes = np.load(labels_classes_path / "generation_classes.npy")

    model_inputs_generator = build_model_inputs(
        path_label_maps=sorted(TestData.get_label_maps()),
        n_labels=len(generation_labels),
        batchsize=1,
        n_channels=1,
        subjects_prob=None,
        generation_classes=generation_classes,
        prior_means=None,
        prior_stds=None,
        prior_distributions="uniform",
        use_specific_stats_for_channel=False,
        mix_prior_and_random=False,
    )

    model_inputs = next(model_inputs_generator)

    return model_inputs

