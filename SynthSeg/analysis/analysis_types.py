import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class FreeSurferLUTEntry:
    """
    Singe entry from the FreeSurfer color map
    """

    name: str = "Unknown"
    """The official name of the region according to FSL"""

    label: int = 0
    """The label or integer for the region in a segmentation"""

    rgba: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    """The color the region should be displayed in according to FSL"""


@dataclass
class TissueType:
    """
    Configuration for the synthesis of a single tissue type.
    It contains all information about how one tissue type is
    synthesized in the fake MRI images.
    """

    label: FreeSurferLUTEntry = FreeSurferLUTEntry()
    segmentation_class: int = 0
    mean_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    stddev_range: List[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class Options:
    """
    Commandline options for the MPM analysis
    """

    label_file: str = None
    """
    The segmentation image with regions and labels according to FSL.
    """

    contrast_files: List[str] = field(default_factory=list)
    """A list of n specific images with different contrast that are analysed.
     The label_file must be a segmentation of these images."""

    rescale_contrasts: bool = True

    clip_min: List[float] = field(default_factory=list)
    """A list of n minimum clip values for each provided contrast image."""

    clip_max: List[float] = field(default_factory=list)
    """A list of n maximum clip values for each provided contrast image."""

    output_dir: str = None
    """Output directory for the analysis result and the rescaled contrast image."""

    equalize_left_right_regions: bool = False
    """
    If true then equivalent regions in the left/right hemisphere will use the same gray-value distributions, 
    even if they are different in the measured contrast images. 
    """

    undefined_region_as_background: bool = True
    """
    If set to True, regions that where not found in the analysed segmentation will not be included in the training.
    If set to False, each undefined region will have a random gray-level distribution in the synthetic map defined
    by the undefined_region_stats option.
    However, these unknown labels will not be used for calculating the loss of the segmentation.
    """

    undefined_region_stats: List[float] = field(default_factory=lambda: [0.1, 0.9, 0.01, 0.1])
    """
    Labels that are required for training but could not be found in the analysed scans need gray-level distributions
    as well.
    Each of these regions will get a separate gray-level distribution, randomly drawn from the provided parameters
    which are in the form [min_mean, max_mean, min_stddev, max_stddev].
    """

    template_generator_config: str = ""
    """
    Provides a template generator configuration that is used. All values are preserved except the ones necessary for
    setting the segmentation labels and gray-value statistics.
    """


@dataclass
class StatisticsOptions:
    """
    Options for specifying how exactly each labeled region is analyzed and what measures are used to estimate
    its Gaussian gray-level distribution.
    """

    method: str = "winsorized"
    """
    Defines how to calculate min/max for the Gaussian gray-level distribution from the data of each region.
    Each method calculates an estimate for the mean and standard deviation of the data.
    Possible values are "winsorized", "median", and "gaussian".
    Method "winsorized" first filters the data by removing outliers up until a specified percentile and then calculates
    the normal mean and standard deviation.
    Method "median" calculates the median of the data and uses the interquartile range with of a specific percentile
    to determine the standard deviation.
    Method "gaussian" calculates the mean and standard deviation on the unfiltered region data. This might lead to bad
    estimates if the data contains outliers or artefacts.
    """

    parameter: float = 0.1
    """
    Parameters for the different methods.
    For method "winsorized", a value specifying the percentages to cut on each side of the data,
    with respect to the number of unmasked data, as float between 0.0 and 0.5 Default 0.1.
    For method "median", the percentile to use for the IQR.
    For method "gaussian", this value is unused.
    """

    range_brackets: List[float] = field(default_factory=lambda: [0.95, 1.05, 0.5, 1.05])
    """
    Factors to defines the range of mean and standard deviation.
    For a setting [m_min, m_max, s_min, s_max], the ranges for the generator are computed like this:
    [m_min*mean, m_max*mean] and [s_min*stddev, s_max*stddev].
    """