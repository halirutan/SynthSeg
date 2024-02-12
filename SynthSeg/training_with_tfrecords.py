import os
from dataclasses import asdict
from typing import Tuple, Union, Optional
from contextlib import nullcontext
from inspect import getmembers, isclass
from pathlib import Path
import numpy as np
from time import perf_counter
import pandas as pd
import glob
import tensorflow as tf

from ext.lab2im import layers
from ext.neuron import layers as nrn_layers
from ext.neuron import models as nrn_models
from . import segmentation_model

from .metrics_model import WeightedL2Loss, DiceLoss, IdentityLoss
from .training_options import TrainingOptions
from .brain_generator import read_tfrecords


class NullStrategy:
    @staticmethod
    def scope():
        return nullcontext()


def training(opts: TrainingOptions) -> tf.keras.callbacks.History:
    """Train the U-net with a TFRecord Dataset.

    Args:
        opts: The training options. The parameters related to the generation of the synthetic images will be ignored.
    """
    # Check epochs
    print(f"Setting seed to {opts.seed}")
    tf.keras.utils.set_random_seed(seed=opts.seed)
    

    assert (opts.wl2_epochs > 0) | (
        opts.dice_epochs > 0
    ), "either wl2_epochs or dice_epochs must be positive, had {0} and {1}".format(
        opts.wl2_epochs, opts.dice_epochs
    )

    if opts.mixed_precision is True:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Define distributed strategy
    if opts.strategy == "null":
        strategy = NullStrategy()
    elif opts.strategy == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise NotImplementedError(f"The '{opts.strategy}' strategy is not implemented.")

    output_dir = Path(opts.model_dir)

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset from tfrecords
    files = sorted(list(Path(opts.tfrecords_dir).glob("*.tfrecord")))
    dataset = read_tfrecords(
        files,
        compression_type=opts.compression_type,
        num_parallel_reads=opts.num_parallel_reads,
    )
    dataset = dataset.batch(opts.batchsize).prefetch(1)

    input_shape = opts.input_shape
    if isinstance(input_shape, int):
        input_shape = (input_shape, input_shape, input_shape, 1)

    checkpoint = opts.checkpoint
    find_last_checkpoint = opts.find_last_checkpoint
    load_after_w2l_pretraining = False
 
    # Define and compile model
        
    with strategy.scope():
        # prepare the segmentation model
        if opts.use_original_unet:
            unet_model = nrn_models.unet(
                input_shape=input_shape,
                nb_labels=opts.n_labels,
                nb_levels=opts.n_levels,
                nb_conv_per_level=opts.nb_conv_per_level,
                conv_size=opts.conv_size,
                nb_features=opts.unet_feat_count,
                feat_mult=opts.feat_multiplier,
                activation=opts.activation,
                batch_norm=-1,
                name="unet",
            )
            
        else:
            unet_model = segmentation_model.unet(
                input_shape=input_shape,
                n_labels=opts.n_labels,
                unet_feat_count=opts.unet_feat_count,
                conv_size=opts.conv_size,
                n_levels=opts.n_levels,
                nb_conv_per_level=opts.nb_conv_per_level,
                activation=opts.activation,
            )

        # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
        if opts.wl2_epochs > 0:
            wl2_model = tf.keras.models.Model(
                unet_model.inputs, [unet_model.get_layer("unet_likelihood").output]
            )
            
            wl2_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=opts.lr),
                loss=WeightedL2Loss(n_labels=opts.n_labels),
            )
            

    results = None
    # additionally check that chackpoint is None, otherwise we assume that we want to
    # resume from dice checkpoint and thus skip w2l-pretraining

    if opts.wl2_epochs > 0 and checkpoint is None:
        
        callbacks = build_callbacks(
            output_dir=output_dir,
            metric_type="wl2",
            wandb=opts.wandb,
            wandb_log_freq=opts.wandb_log_freq,
            training_opts=opts,
        )

        # fit
        results = wl2_model.fit(
            dataset,
            epochs=opts.wl2_epochs,
            steps_per_epoch=opts.steps_per_epoch or None,
            callbacks=callbacks,
        )

        print("Number of iterations wl2_model seen: ", wl2_model.optimizer.iterations.numpy())

        if opts.wandb:
            import wandb
            wandb.finish()

        checkpoint = output_dir / ("wl2_epoch-%03d.keras" % opts.wl2_epochs)
        find_last_checkpoint = False
        load_after_w2l_pretraining = True

    if opts.dice_epochs > 0:
        with strategy.scope():
            # fine-tuning with dice metric
            dice_model, is_compiled, init_epoch, init_batch = load_model(
                model=unet_model, checkpoint=checkpoint, metric_type="dice", 
                find_last_checkpoint = find_last_checkpoint,
                load_after_w2l_pretraining = load_after_w2l_pretraining
            )

            if not is_compiled:
                dice_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=opts.lr),
                    loss=DiceLoss(n_labels=opts.n_labels),
                )

        print(f"Optimizers  number of iterations at the beginning: ",  dice_model.optimizer.iterations.numpy())
        print(f"Amount of batches that will be skipped: {opts.wl2_epochs*opts.steps_per_epoch} batches from wl2 pretraining + {init_batch} from training on dice loss")
        print(f"Restarting from heckpoint: {init_epoch}")
        dataset = dataset.skip(init_batch+opts.wl2_epochs*opts.steps_per_epoch)
        
        callbacks = build_callbacks(
            output_dir=output_dir,
            metric_type="dice",
            wandb=opts.wandb,
            wandb_log_freq=opts.wandb_log_freq,
            training_opts=opts,
        )

        results = dice_model.fit(
            dataset,
            epochs=opts.dice_epochs,
            steps_per_epoch=opts.steps_per_epoch or None,
            callbacks=callbacks,
            initial_epoch=init_epoch,
        )

    return results


def load_model(
    model: tf.keras.models.Model,
    checkpoint: Path,
    metric_type: str,
    reinitialise_momentum: bool = False,
    find_last_checkpoint: bool = True, 
    load_after_w2l_pretraining: bool = False
) -> Tuple[tf.keras.models.Model, bool, int]:
    is_compiled = False
    init_epoch = 0
    init_batch_idx = 0

    def find_last_ckpt(checkpoint, metric_type):
        files = [el for el in glob.glob(f"{checkpoint}/{metric_type}*.keras", recursive=True)]
        assert len(files)>0, f"Trying to load model for continuing training with {metric_type} loss, but suitable files were found"
        files_pd = pd.DataFrame.from_records([
            dict(
            fullpath=el,
            ckpt=int(str(Path(el).name).split("epoch-")[1].split(".keras")[0])
            ) for el in files])
        checkpoint = Path(max(list(files_pd[files_pd.ckpt==files_pd.ckpt.max()].fullpath), key=os.path.getctime)) 
        return checkpoint

    if checkpoint is not None:
        if find_last_checkpoint:
            checkpoint = find_last_ckpt(checkpoint, metric_type)
            print(f"Model will continue from  {checkpoint}")
        else: 
            assert str(checkpoint).endswith(".keras"), f"Path provided to checkpoint file doesnt have '.keras' extension!"
            checkpoint = Path(checkpoint)
            assert checkpoint.exists(), f"Trying to load model for continuing training, but path {checkpoint} doesn't exits"

        if not load_after_w2l_pretraining:
            assert metric_type in checkpoint.name, f"Trying to load model for continuing training with {metric_type} loss, but None were found"
            init_epoch = int(str(checkpoint.name).split("epoch-")[1].split(".keras")[0])
        
        if (not reinitialise_momentum) & (metric_type in checkpoint.name):
            print("loading model with optimizer states.")

            custom_l2i = {
                key: value
                for (key, value) in getmembers(layers, isclass)
                if key != "Layer"
            }
            custom_nrn = {
                key: value
                for (key, value) in getmembers(nrn_layers, isclass)
                if key != "Layer"
            }

            custom_objects = {
                **custom_l2i,
                **custom_nrn,
                "tf": tf,
                "keras": tf.keras,
                "loss": IdentityLoss().loss,
                "DiceLoss": DiceLoss, 
                "WeightedL2Loss": WeightedL2Loss

            }
            model = tf.keras.models.load_model(
                checkpoint, custom_objects=custom_objects
            )
            init_batch_idx = model.optimizer.iterations.numpy()

            is_compiled = True
        else:
            model.load_weights(checkpoint, by_name=True)
            print("Loading weights only")

    return model, is_compiled, init_epoch, init_batch_idx


def build_callbacks(
    output_dir: Path,
    metric_type,
    wandb: bool = False,
    wandb_log_freq: Union[str, int] = "epoch",
    training_opts: Optional[TrainingOptions] = None,
):
    # create log folder
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # model saving callback
    save_file_name = os.path.join(output_dir, "%s_epoch-{epoch:03d}.keras" % metric_type)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(save_file_name, verbose=1, save_weights_only=False)]

    # TensorBoard callback
    if metric_type == "dice":
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False
            )
        )

    # WandB callback
    if wandb:
        import wandb as wandbm
        from wandb.integration.keras import WandbMetricsLogger

        wandbm.init(config=asdict(training_opts) if training_opts else None)
        callbacks.append(WandbMetricsLogger(log_freq=wandb_log_freq))

    return callbacks
