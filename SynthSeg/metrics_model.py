"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import numpy as np
import tensorflow as tf

# third-party imports
from ext.lab2im import layers


def metrics_model(input_model, label_list, metrics='dice'):

    # get prediction
    last_tensor = input_model.outputs[0]
    input_shape = last_tensor.get_shape().as_list()[1:]

    # check shapes
    n_labels = input_shape[-1]
    label_list = np.unique(label_list)
    assert n_labels == len(label_list), 'label_list should be as long as the posteriors channels'

    # get GT and convert it to probabilistic values
    labels_gt = input_model.get_layer('labels_out').output
    labels_gt = layers.ConvertLabels(label_list)(labels_gt)
    labels_gt = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels_gt)
    labels_gt = tf.keras.layers.Reshape(input_shape)(labels_gt)

    # make sure the tensors have the right keras shape
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    labels_gt._keras_shape = tuple(labels_gt.get_shape().as_list())

    if metrics == 'dice':
        last_tensor = layers.DiceLoss()([labels_gt, last_tensor])

    elif metrics == 'wl2':
        last_tensor = layers.WeightedL2Loss(target_value=5)([labels_gt, last_tensor])

    else:
        raise Exception('metrics should either be "dice or "wl2, got {}'.format(metrics))

    # create the model and return
    model = tf.keras.models.Model(inputs=input_model.inputs, outputs=last_tensor)
    return model


class IdentityLoss(object):
    """Very simple loss, as the computation of the loss as been directly implemented in the model."""
    def __init__(self, keepdims=True):
        self.keepdims = keepdims

    def loss(self, y_true, y_predicted):
        """Because the metrics is already calculated in the model, we simply return y_predicted.
           We still need to put y_true in the inputs, as it's expected by keras."""
        loss = y_predicted

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss


class WeightedL2Loss(tf.keras.losses.Loss):
    def __init__(
        self,
        n_labels: int,
        target_value: float = 5.0,
        background_weight: float = 1.0e-4,
    ):
        super().__init__()
        self._n_labels = n_labels
        self._target_value = target_value
        self._background_weight = background_weight

    def get_config(self):
        config = {
            "n_labels": self._n_labels,
            "target_value": self._target_value,
            "background_weight": self._background_weight,
        }
        return config

    def __call__(self, gt, pred, **kwargs):
        pred = tf.cast(pred, tf.float32)
        gt = tf.one_hot(gt, depth=self._n_labels, axis=-1, dtype=pred.dtype)
        weights = tf.expand_dims(1 - gt[..., 0] + self._background_weight, -1)
        return tf.keras.backend.sum(
            weights * tf.keras.backend.square(pred - self._target_value * (2 * gt - 1))
        ) / (tf.keras.backend.sum(weights) * self._n_labels)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, n_labels: int, dim: int = 3, enable_checks: bool = True):
        super().__init__()
        self._n_labels = n_labels
        self._dim = dim
        self._enable_checks = enable_checks

    def get_config(self):
        config = {
            "n_labels": self._n_labels,
            "dim": self._dim,
            "enable_checks": self._enable_checks,
        }
        return config

    def __call__(self, gt, pred, **kwargs):
        pred = tf.cast(pred, tf.float32)
        gt = tf.one_hot(gt, depth=self._n_labels, axis=-1, dtype=pred.dtype)
        # make sure tensors are probabilistic
        if (
            self._enable_checks
        ):  # disabling is useful to, e.g., use incomplete label maps
            gt = tf.keras.backend.clip(
                gt
                / (
                    tf.math.reduce_sum(gt, axis=-1, keepdims=True)
                    + tf.keras.backend.epsilon()
                ),
                0,
                1,
            )
            pred = tf.keras.backend.clip(
                pred
                / (
                    tf.math.reduce_sum(pred, axis=-1, keepdims=True)
                    + tf.keras.backend.epsilon()
                ),
                0,
                1,
            )

        # compute dice loss for each label
        top = tf.math.reduce_sum(2 * gt * pred, axis=list(range(1, self._dim + 1)))
        bottom = tf.math.square(gt) + tf.math.square(pred) + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, self._dim + 1)))
        last_tensor = top / bottom

        return tf.keras.backend.mean(1 - last_tensor)


class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        We need a custom implementation, since the shape of y_true
        does not seem to be known when using TFRecordDatasets ...

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        if not self.sparse_y_true:
            y_true = tf.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = tf.argmax(y_pred, axis=self.axis)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        #if y_true.shape.ndims > 1:
        y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        if self.ignore_class is not None:
            ignore_class = tf.cast(self.ignore_class, y_true.dtype)
            valid_mask = tf.not_equal(y_true, ignore_class)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[valid_mask]

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=self._dtype,
        )
        return self.total_cm.assign_add(current_cm)