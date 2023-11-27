import gc
import os
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List
from simple_parsing import ArgumentParser

from SynthSeg.training_options import TrainingOptions
from ext.neuron import models as nrn_models
from SynthSeg.logging_utils import get_logger
from SynthSeg.option_utils import get_absolute_path

logger = get_logger(os.path.basename(__file__))


@dataclass
class Options:
    """
    Commandline options for specifying the UNet
    """

    config: Optional[str] = None
    """Path to a training config file that specifies the parameters."""

    net_sizes: List[int] = field(default_factory=lambda: [256])

    bits: int = 32
    """
    Bit size of the parameters in the neural network. Defaults to 32 for type float32 but other settings are possible.
    """


def main():
    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(Options, "general")
    args = parser.parse_args()
    arguments: Options = args.general

    if not isinstance(arguments.config, str):
        logger.error("You must specify a valid path to a training configuration.")
        exit(1)

    conf_file = get_absolute_path(arguments.config)
    if isinstance(conf_file, str) and os.path.isfile(conf_file) and os.access(conf_file, os.R_OK):
        logger.info("Loading training config from configuration file.")

        # Load config and make paths within the config absolute
        opts = TrainingOptions.load(conf_file)
        opts = (opts
                .with_absolute_paths(os.path.abspath(conf_file))
                .convert_lists_to_numpy()
                )
    else:
        logger.error(f"Unable to load training config from file {conf_file}")
        exit(1)

    for s in arguments.net_sizes:
        print(f"Size: {s}")
        unet_model = nrn_models.unet(
            input_shape=[s, s, s, 3],
            nb_labels=opts.n_labels,
            nb_levels=opts.n_levels,
            nb_conv_per_level=opts.nb_conv_per_level,
            conv_size=opts.conv_size,
            nb_features=opts.unet_feat_count,
            feat_mult=opts.feat_multiplier,
            activation=opts.activation,
            batch_norm=-1
        )
        print(unet_model.summary())
        unet_model.save("/home/patrick/tmp/unet")
        parameter_count = unet_model.count_params()
        bit_size = parameter_count * arguments.bits
        megabytes = np.ceil(bit_size / (8 * 2**20))
        logger.info(f"Megabytes: {megabytes}")

        del unet_model
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == '__main__':
    main()
