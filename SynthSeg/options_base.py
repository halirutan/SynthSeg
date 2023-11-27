import numpy as np
from typing import List

from SynthSeg.option_utils import get_absolute_path


class OptionsBase:
    """
    Base class for options. Derived classes must provide overrides for
    `get_np_list_options` and `get_non_path_string_options`!
    """

    @staticmethod
    def get_np_list_options() -> List[str]:
        """
        Provides names of attributes that should be converted to numpy arrays.
        This method should be overriden in the derived class.
        """
        return []

    @staticmethod
    def get_non_path_string_options() -> List[str]:
        """
        Provides names of attributes that are strings, but are not paths.
        These attributes will not be converted to absolute paths.
        """
        return []

    def with_absolute_paths(self, reference_file: str):
        """
        Converts path attributes to absolute paths using a reference file.

        Args:
            reference_file (str): The reference file to be used for generating absolute paths.

        Returns:
            A copy of the object with absolute paths added.
        """
        copy = self.__class__()
        for key, value in vars(self).items():
            if isinstance(value, str) and key not in self.__class__.get_non_path_string_options():
                setattr(copy, key, get_absolute_path(value, reference_file))
            else:
                setattr(copy, key, value)
        return copy

    def convert_lists_to_numpy(self):
        """
        Converts specific options from normal lists to numpy arrays.

        Returns:
            A copy of the instance with number lists converted to numpy arrays.
        """
        copy = self.__class__()
        for key, value in vars(self).items():
            if key in self.__class__.get_np_list_options() and isinstance(value, list):
                setattr(copy, key, np.array(value))
            else:
                setattr(copy, key, value)
        return copy
