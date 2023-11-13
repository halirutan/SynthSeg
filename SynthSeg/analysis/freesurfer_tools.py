import numpy as np
import os
import re

from SynthSeg.analysis.analysis_types import FreeSurferLUTEntry

def get_free_surfer_lut() -> dict:
    """
    Provides all default labels of FreeSurfer as a dictionary.

    Returns:
        dict: Mapping of label id to `FreeSurferLUTEntry`
    """
    lut_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freeSurferLUT.txt")
    assert os.path.isfile(lut_file), f"Tried to find file {lut_file}"
    result = {}
    with open(lut_file) as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0 or stripped[0] == "#":
            continue

        parts = stripped.split()
        assert len(parts) == 6

        label_id = int(parts[0])
        name = parts[1]
        rgba = np.array([
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5])
        ])

        assert label_id not in result
        result[label_id] = FreeSurferLUTEntry(name, label_id, rgba)
    return result


# Global variable to hold the lookup table without reloading and parsing the file over and over again.
FSL_LUT = get_free_surfer_lut()

# Global regexes that match side-specific regions in the label names of the FSL lookup table
FSL_LEFT_LABEL_REGEX = re.compile(r"^([Ll])eft[_-]|^ctx-(lh)-|^wm[_-](lh)[_-]|(_l)$")
FSL_RIGHT_LABEL_REGEX = re.compile(r"^([Rr])ight[_-]|^ctx-(rh)-|^wm[_-](rh)[_-]|(_r)$")


def substitute_left_right(match) -> str:
    """
    Substitution from a left label name of an FSL region to its right-sided counterpart.
    For a usage example, see test_fsl_tools.py

    Args:
        match: A Match object representing the match found in the string.

    Returns:
        str: The modified string after substituting the left specific portion with its right counterpart.

    """
    if match.group(1):
        if match.group(1) == "L":
            return match.group().replace("Left", "Right")
        else:
            return match.group().replace("left", "right")
    elif match.group(2) or match.group(3):
        return match.group().replace("lh", "rh")
    elif match.group(4):
        return match.group().replace("_l", "_r")
    else:
        return match.group()


def substitute_right_left(match) -> str:
    """
    See doc of `substitute_left_right`
    Args:
        match: A `re.Match` object representing the matched pattern.

    Returns:
        A string with the substitutions made based on the match group.

    """
    if match.group(1):
        if match.group(1) == "R":
            return match.group().replace("Right", "Left")
        else:
            return match.group().replace("right", "left")
    elif match.group(2) or match.group(3):
        return match.group().replace("rh", "lh").replace("rh", "lh")
    elif match.group(4):
        return match.group().replace("_r", "_l")
    else:
        return match.group()
