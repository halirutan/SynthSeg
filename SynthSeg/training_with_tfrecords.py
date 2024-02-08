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
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

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

    num_gpu  = strategy.num_replicas_in_sync if opts.strategy != "none" else len(tf.config.list_physical_devices('GPU'))
    
    if opts.scaling_type == "strong" and strategy is not None:
            steps_per_epoch  = opts.steps_per_epoch//num_gpu
            print(f"Number of steps per epoch was scaled to {opts.steps_per_epoch}")
    elif opts.scaling_type ==" weak" and strategy is not None: 
        steps_per_epoch = opts.steps_per_epoch
    else: 
        steps_per_epoch = opts.steps_per_epoch

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
    if opts.wl2_epochs > 0:
        callbacks = build_callbacks(
            output_dir=output_dir,
            metric_type="wl2",
            wandb=opts.wandb,
            wandb_log_freq=opts.wandb_log_freq,
            training_opts=opts,
            num_gpu =num_gpu, 
            steps_per_epoch=steps_per_epoch

        )

        # fit
        results = wl2_model.fit(
            dataset,
            epochs=opts.wl2_epochs,
            steps_per_epoch=steps_per_epoch or None,
            callbacks=callbacks,
            
        )

        if opts.wandb:
            import wandb

            wandb.finish()

        checkpoint = output_dir / ("wl2_%03d.keras" % opts.wl2_epochs)
        find_last_checkpoint = False


    if opts.dice_epochs > 0:
        
        with strategy.scope():
            # fine-tuning with dice metric
            dice_model, is_compiled, init_epoch, seen_samples = load_model(
                model=unet_model, checkpoint=checkpoint, metric_type="dice", find_last_checkpoint = find_last_checkpoint ,
                reinitialise_momentum = opts.save_weights_only
            )
            if not is_compiled:
                dice_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=opts.lr),
                    loss=DiceLoss(n_labels=opts.n_labels),
                )

        if seen_samples>0:
            # Unbatching and batching back in done for the case when batch size changes for some reason: e.g. num_gpu changed 
            dataset= dataset.unbatch()
            print(f"Amount of samples that will be skipped: {seen_samples}")
            dataset = dataset.skip(seen_samples).batch(opts.batchsize).prefetch(1)
        
        #ToDo: remove print later
        print(f"Last layer of optimizer: ",  dice_model.optimizer.variables[-1].numpy())
        print(f"Optimizers  number of iterations at the beginning: ",  dice_model.optimizer.iterations.numpy())
      
        
        callbacks = build_callbacks(
            output_dir=output_dir,
            metric_type="dice",
            wandb=opts.wandb,
            wandb_log_freq=opts.wandb_log_freq,
            training_opts=opts,
            num_gpu =num_gpu, 
            steps_per_epoch=steps_per_epoch
        )

        results = dice_model.fit(
            dataset,
            epochs=opts.dice_epochs,
            steps_per_epoch=steps_per_epoch or None,
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
) -> Tuple[tf.keras.models.Model, bool, int]:
    is_compiled = False
    init_epoch = 0
    seen_samples = 0

    if checkpoint is not None:
        if find_last_checkpoint:
            files = pd.DataFrame.from_records([dict(fullpath=el, ckpt=int(str(Path(el).name).split(metric_type)[1].split(".keras")[0][1:]))
                                                for el in glob.glob(f"{checkpoint}/*.keras", recursive=True)])
            checkpoint = Path(max(list(files[files.ckpt==files.ckpt.max()].fullpath), key=os.path.getctime)) 
            print(f"Model will continue from  {checkpoint}")
        else:
            checkpoint = Path(checkpoint)
            
        # if metric_type in checkpoint.name:
        init_epoch = int(str(checkpoint.name).split("epoch-")[1].split("_seen_samples")[0])
        seen_samples =  int(str(checkpoint.name).split("_seen_samples-")[1].split(".keras")[0])

        if (not reinitialise_momentum) & (metric_type in checkpoint.name):
            print("loading model with states ")

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
                "DiceLoss": DiceLoss
            }
            model = tf.keras.models.load_model(
                checkpoint, custom_objects=custom_objects
            )
            # init_batch_idx = model.optimizer.iterations.numpy()

            is_compiled = True
        else:
            model.load_weights(checkpoint, by_name=True)
            print("loading weights only")


    return model, is_compiled, init_epoch, seen_samples


     
class ModelCheckpointCustom(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                batch_size,
                 filepath,
                 monitor: str = "val_loss",
                 verbose: int = 0,
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 mode: str = "auto",
                 save_freq="epoch",
                 options=None,
                 initial_value_threshold=None,
                 **kwargs):

        super().__init__(
            filepath = filepath,
            monitor=monitor,
        verbose=verbose,
        save_best_only = save_best_only,
        save_weights_only = save_weights_only,
        mode=mode,
        save_freq = save_freq,
        options = options,
        initial_value_threshold = initial_value_threshold,
        **kwargs)

        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs["seen_samples"] = self.model.optimizer.iterations.numpy()*self.batch_size
        super().on_epoch_end(epoch, logs)

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename=None, append = True):
        super().__init__()
        self.filename = filename
        # super().__init__(filename=filename, append = append)
        self.tracked_time = []
        print("Created Timing callback, results will be saved under ", self.filename)
        self._chief_worker_only = True

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_begin_time = perf_counter()
        self.bacth_end_time = None

    def on_train_batch_end(self, batch, logs=None):
        batch_end_time = perf_counter()
        time_elapsed_for_batch = batch_end_time - self.batch_begin_time
        replica_context = tf.distribute.get_replica_context() 
        replica_id = replica_context.replica_id_in_sync_group  if replica_context is not None else None
        
        self.tracked_time.append({"type": "batch", "epoch_idx": self.current_epoch, "batch_idx": batch,
                                  "begin_time": self.batch_begin_time,
                                  "end_time": batch_end_time,
                                  "time_elapsed": time_elapsed_for_batch,
                                  "replica_id":replica_id})

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_time = perf_counter()
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = perf_counter()
        time_elapsed_for_epoch = epoch_end_time - self.epoch_begin_time
        replica_context = tf.distribute.get_replica_context() 
        replica_id = replica_context.replica_id_in_sync_group  if replica_context is not None else None
        
        self.tracked_time.append({"type": "epoch", "epoch_idx": epoch,
                                  "begin_time": self.epoch_begin_time,
                                  "end_time": epoch_end_time,
                                  "time_elapsed": time_elapsed_for_epoch,
                                    "replica_id":replica_id })

        print(f"Last layer of optimizer at the end of epoch {epoch}: ",  self.model.optimizer.variables[-1].numpy())
        print(f"Optimizers  number of iterations at the end of epoch {epoch}: ",  self.model.optimizer.iterations.numpy())

        # if epoch==1:
        #     ckpt =Path(self.filename).resolve().parent/f"model_epoch-{epoch}.keras"
        #     print(f"saving model additionally to {ckpt}")
        #     self.model.save(str(ckpt))



  

    def on_train_begin(self, logs=None):
        self.train_begin_time = perf_counter()
        # super().on_train_begin(self, logs)


    def on_train_end(self, logs=None):
        train_end_time = perf_counter()
        time_elapsed_for_train = train_end_time - self.train_begin_time
        replica_context = tf.distribute.get_replica_context() 
        replica_id = replica_context.replica_id_in_sync_group  if replica_context is not None else None
        
        self.tracked_time.append({"type": "fit",
                                  "begin_time": self.train_begin_time,
                                  "end_time": train_end_time,
                                  "time_elapsed": time_elapsed_for_train,
                                   "replica_id":replica_id})

        tracked_time_pd = pd.DataFrame.from_records(self.tracked_time)
        tracked_time_pd.to_csv(self.filename, index=False)


def build_callbacks(
    output_dir: Path,
    metric_type,
    wandb: bool = False,
    wandb_log_freq: Union[str, int] = "epoch",
    training_opts: Optional[TrainingOptions] = None,
    num_gpu: int = None, 
    steps_per_epoch: int = None
):
    # create log folder
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # model saving callback
    save_file_name = os.path.join(output_dir, "%s_epoch-{epoch:03d}_seen_samples-{seen_samples}.keras" % metric_type)
    # save_file_name_h5 = os.path.join(output_dir, "%s_{epoch:03d}.h5" % metric_type)


    tracking_file = os.path.join(output_dir, f"time_recorder-num_gpu_{num_gpu}-strategy-{training_opts.strategy}_scaling-{training_opts.scaling_type}_steps_per_epoch-{steps_per_epoch}.csv")
    tracking_callback = TimingCallback(tracking_file)
    # save_weights_only=True is a workaround when training with mixed precision: https://github.com/keras-team/tf-keras/issues/203
    # For now, setting this to true disables the posibility to train a model further, at least if we dont want to reinit the momentum ...
    print(f"Model checkpoint save_weights_only?: {training_opts.save_weights_only}")

    callbacks = [tf.keras.callbacks.ModelCheckpointCustom(batch_size = training_opts.batchsize, filepath =save_file_name, verbose=1, save_weights_only=training_opts.save_weights_only), 
                #  tf.keras.callbacks.ModelCheckpoint(save_file_name_h5, verbose=1, save_weights_only=training_opts.save_weights_only), 
                 tracking_callback]

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
