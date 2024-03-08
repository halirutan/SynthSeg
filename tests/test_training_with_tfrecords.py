import tensorflow as tf
import numpy as np
import pytest

from SynthSeg.training_options import TrainingOptions
from SynthSeg.training_with_tfrecords import training


@pytest.mark.parametrize(
    "wl2_epochs, dice_epochs, mean, std, exact, files",
    [
        (
            1,
            0,
            22.4182,
            0.3077,
            21.9824,
            ["wl2_epoch-001.keras"],
        ),
        (
            0,
            1,
            0.961943,
            0.003861203,
            0.964388,
            ["dice_epoch-001.keras"],
        ),
        (
            1,
            1,
            0.972157,
            0.0021950396925116346,
            0.972157,
            ["wl2_epoch-001.keras", "dice_epoch-001.keras"],
        ),
    ],
    ids=["wl2", "dice", "dice_after_wl2"],
)
def test_training(tmp_path, tfrecord, wl2_epochs, dice_epochs, mean, std, exact, files):
    """
    Tests the equivalence with the original training via `mean` and `std`
    and the current implementation via `exact`.
    """
    tf.keras.utils.set_random_seed(43)

    opts = TrainingOptions(
        model_dir=str(tmp_path / "output"),
        wl2_epochs=wl2_epochs,
        dice_epochs=dice_epochs,
        steps_per_epoch=int(tfrecord.num_samples / 2)
        if wl2_epochs + dice_epochs == 2
        else tfrecord.num_samples,
        batchsize=1,
        tfrecords_dir=str(tfrecord.path.parent),
        valid_tfrecords_dir=str(tfrecord.path.parent),
        input_shape=list(tfrecord.shape),
        n_labels=tfrecord.n_labels,
    )

    results = training(opts)
    output_files = [p.name for p in (tmp_path / "output").iterdir()]

    # mean and std were obtained via the experiment below
    np.testing.assert_allclose(results.history["loss"][0], mean, atol=2 * std)
    np.testing.assert_allclose(results.history["loss"][0], exact, rtol=1e-5)
    assert all([f in output_files for f in files])

    # Experiment (remove the tfrecord fixture):
    # from SynthSeg.training import training_from_options
    # from SynthSeg.training_options import TrainingOptions
    # from . import TestData
    #
    # opts = TrainingOptions(
    #     labels_dir=TestData.get_label_maps()[0],
    #     target_res=8,
    #     model_dir=str(tmp_path / "output"),
    #     wl2_epochs=wl2_epochs,
    #     dice_epochs=dice_epochs,
    #     steps_per_epoch=10,
    #     batchsize=1,
    # )
    # losses = np.empty(10)
    # for i in range(10):
    #     results = training_from_options(opts)
    #     print(results.history)
    #     losses[i] = results.history["loss"][-1]
    # print(losses, losses.mean(), losses.std())
