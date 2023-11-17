import numpy as np
import os
from typing import Type

from SynthSeg.brain_generator_options import GeneratorOptions
from SynthSeg.options_base import OptionsBase
from SynthSeg.training_options import TrainingOptions


def test_numpy_replacement():
    """
    Check if the conversion from lists to numpy arrays works correctly.
    We check both option classes and go through all the possible list types,
    then we convert them to numpy arrays and check their type.
    """
    run_numpy_replacement(GeneratorOptions)
    run_numpy_replacement(TrainingOptions)


def test_path_replacement():
    """
    We check all string fields that are not explicitly excluded.
    However, we only check if the original paths from the default instantiation are
    relative (which it is not required to be) and if they are absolute after conversion.
    """
    run_path_replacement(GeneratorOptions)
    run_numpy_replacement(TrainingOptions)


def run_numpy_replacement(clazz: Type[OptionsBase]):
    opts = clazz()
    np_opts = opts.convert_lists_to_numpy()
    list_attributes = opts.get_np_list_options()

    for a in list_attributes:
        assert hasattr(opts, a), f"Field {a} does not exist."
        if isinstance(getattr(opts, a), list):
            assert isinstance(getattr(np_opts, a), np.ndarray), f"List {a} is not a numpy array."


def run_path_replacement(clazz: Type[OptionsBase]):
    opts = clazz()
    names = vars(opts).keys()
    abs_opts = opts.with_absolute_paths(__file__)
    excluded_attributes = opts.get_non_path_string_options()

    for n in names:
        assert hasattr(opts, n), f"Field {n} does not exist."
        field = getattr(opts, n)
        if isinstance(field, str) and n not in excluded_attributes:
            path = field
            abs_path = getattr(abs_opts, n)

            assert not os.path.isabs(path), f"Path {n} should be relative"
            assert os.path.isabs(abs_path), f"Path {n} is not absolute"
