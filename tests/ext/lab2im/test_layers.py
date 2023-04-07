from ext.lab2im.layers import RandomCrop
import numpy as np
from keras.models import Model
from keras.layers import Input
from pathlib import Path


def test_random_crop(model_inputs, set_random_seeds):
    random_crop_target_path = Path(__file__).parent / "random_crop_seed43.npy"

    labels_input = Input(shape=model_inputs[0].shape[1:], name='labels_input', dtype='int32')
    random_crop_layer = RandomCrop(crop_shape=[16, 16, 16])(labels_input)

    model = Model(inputs=labels_input, outputs=random_crop_layer)

    out = model.predict(model_inputs[0])

    out_target = np.load(str(random_crop_target_path))

    assert np.allclose(out, out_target)



