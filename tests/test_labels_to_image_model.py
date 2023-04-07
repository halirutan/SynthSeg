import pickle

from SynthSeg.labels_to_image_model import labels_to_image_model
from ext.lab2im import utils
from tests import TestData
from pathlib import Path
import numpy as np


def test_labels_to_image_model(model_inputs, set_random_seeds):
    output_target_path = Path(__file__).parent / "labels_to_image_model_seed43.pkl"
    label_map = sorted(TestData.get_label_maps())[0]

    generation_labels, _ = utils.get_list_labels(labels_dir=label_map)

    model = labels_to_image_model(
        labels_shape=model_inputs[0].shape[1:-1],
        n_channels=1,
        generation_labels=generation_labels,
        output_labels=generation_labels,
        n_neutral_labels=len(generation_labels),
        atlas_res=np.array([1., 1., 1.]),
        target_res=None,
        output_shape=16,
        aff=np.eye(4),
        nonlin_scale=0.04,
        randomise_res=True,
    )

    output = model.predict(model_inputs)

    with output_target_path.open("rb") as f:
        output_target = pickle.load(f)

    for out, tar in zip(output, output_target):
        assert np.allclose(out, tar)
