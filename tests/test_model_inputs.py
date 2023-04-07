from SynthSeg.model_inputs import build_model_inputs
from . import TestData
import numpy as np
from pathlib import Path
import pickle


def test_build_model_inputs():
    # TODO: how should we deal with large files?
    model_inputs_path = Path(__file__).parent / "model_inputs_seed43.pkl"

    labels_classes_path = TestData.synth_seg_path / "data" / "labels_classes_priors"
    generation_labels = np.load(labels_classes_path / "generation_labels.npy")
    generation_classes = np.load(labels_classes_path / "generation_classes.npy")

    np.random.seed(43)

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

    with model_inputs_path.open(mode="br") as f:
        # pickle.dump(model_inputs, f)
        model_inputs_target = pickle.load(f)

    for inp, tar in zip(model_inputs, model_inputs_target):
        assert np.allclose(inp, tar)
