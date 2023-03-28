import numpy as np
import keras.layers as KL
from keras.models import Model
from ext.lab2im import layers
import tensorflow as tf

from SynthSeg.model_inputs import build_model_inputs
from ext.lab2im.utils import draw_value_from_distribution
from . import TestData


def test_draw_value_from_distribution():
    """Have to set seed for np only"""
    np.random.seed(1)
    val1 = draw_value_from_distribution(None)

    np.random.seed(1)
    val2 = draw_value_from_distribution(None)

    assert np.isclose(val1, val2)


def test_random_spatial_deformation():
    """Fails!"""
    label_map_files = TestData.get_label_maps()

    batchsize = 1
    n_channels = 1
    generation_labels = np.array(range(20))
    subjects_prob = None
    generation_classes = np.array(range(20))
    prior_means = None
    prior_stds = None
    prior_distributions = "uniform"
    use_specific_stats_for_channel = False
    mix_prior_and_random = False

    model_inputs_generator = build_model_inputs(path_label_maps=label_map_files,
                                                n_labels=len(generation_labels),
                                                batchsize=batchsize,
                                                n_channels=n_channels,
                                                subjects_prob=subjects_prob,
                                                generation_classes=generation_classes,
                                                prior_means=prior_means,
                                                prior_stds=prior_stds,
                                                prior_distributions=prior_distributions,
                                                use_specific_stats_for_channel=use_specific_stats_for_channel,
                                                mix_prior_and_random=mix_prior_and_random)

    labels_shape = [256, 256, 256]
    scaling_bounds = 0.2
    rotation_bounds = 15
    shearing_bounds = 0.012
    translation_bounds = False
    nonlin_std = 3.0
    nonlin_scale = 0.04

    # define model inputs
    labels_input = KL.Input(shape=labels_shape + [1], name='labels_input', dtype='int32')
    means_input = KL.Input(shape=list(generation_labels.shape) + [n_channels], name='means_input')
    stds_input = KL.Input(shape=list(generation_labels.shape) + [n_channels], name='std_devs_input')
    list_inputs = [labels_input, means_input, stds_input]

    # deform labels
    labels = layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                             rotation_bounds=rotation_bounds,
                                             shearing_bounds=shearing_bounds,
                                             translation_bounds=translation_bounds,
                                             nonlin_std=nonlin_std,
                                             nonlin_scale=nonlin_scale,
                                             inter_method='nearest')(labels_input)

    label_model = Model(inputs=list_inputs, outputs=labels)

    model_inputs = next(model_inputs_generator)

    np.random.seed(1)
    tf.random.set_seed(1)
    labels = label_model.predict(model_inputs)

    np.random.seed(1)
    tf.random.set_seed(1)
    labels2 = label_model.predict(model_inputs)

    assert np.isclose(labels, labels2).all()

