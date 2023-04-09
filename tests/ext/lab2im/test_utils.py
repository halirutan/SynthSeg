from ext.lab2im import utils
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
import numpy as np


def test_draw_value_from_distribution(set_random_seeds):
    """Have to set seed for np only"""
    batch_size = Input(batch_shape=(1,), dtype="int32")

    val = utils.draw_value_from_distribution(
        None, size=3, return_as_tensor=True, batchsize=batch_size
    )
    model = Model(inputs=batch_size, outputs=val)
    output = model.predict(tf.constant([0], dtype="int32"), steps=1)

    assert np.allclose(output, np.array([9.325514, 7.4239902, -6.642418]))
