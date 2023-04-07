from ext.lab2im import layers
import numpy as np
from keras.models import Model
from keras.layers import Input
from pathlib import Path


def test_random_crop(model_inputs, set_random_seeds):
    random_crop_target_path = Path(__file__).parent / "random_crop_seed43.npy"

    labels_input = Input(shape=model_inputs[0].shape[1:], name='labels_input', dtype='int32')
    random_crop_layer = layers.RandomCrop(crop_shape=[16, 16, 16])(labels_input)

    model = Model(inputs=labels_input, outputs=random_crop_layer)

    out = model.predict(model_inputs[0])

    out_target = np.load(str(random_crop_target_path))

    assert np.allclose(out, out_target)


def test_random_spatial_deformation(model_inputs, set_random_seeds):
    random_spatial_deformation_target_path = Path(__file__).parent / "random_spatial_deformation_seed43.npy"

    labels_input = Input(shape=model_inputs[0].shape[1:], name='labels_input', dtype='int32')

    random_spatial_deformation_layer = layers.RandomSpatialDeformation(
        scaling_bounds=0.2,
        rotation_bounds=15,
        shearing_bounds=0.012,
        nonlin_std=3.0,
        nonlin_scale=0.04,
        inter_method="nearest",
    )(labels_input)

    model = Model(inputs=labels_input, outputs=random_spatial_deformation_layer)

    output = model.predict(model_inputs[0])

    output_target = np.load(str(random_spatial_deformation_target_path))

    assert np.allclose(output, output_target)




