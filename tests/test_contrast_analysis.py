import numpy as np

from SynthSeg.analysis.contrast_analysis import generate_tissue_types_from_sample
from SynthSeg.analysis.analysis_types import StatisticsOptions


def test_generate_tissue_types_from_sample():
    mean = 128.0
    std_dev = 32.0
    shape = (100, 100, 100)
    contrast_data = np.random.normal(mean, std_dev, shape)

    # Change 5% of the contrast_data to low values and 5% to high values to simulate high and low
    # artifacts we will encounter.
    # This should give good results for methods "winsoring" and "median", but skewed results for "gaussian".
    num_elements = contrast_data.size
    num_change = int(num_elements * 0.05)
    indices_low = np.unravel_index(np.random.choice(num_elements, num_change, replace=False), shape)
    indices_high = np.unravel_index(np.random.choice(num_elements, num_change, replace=False), shape)

    contrast_data[indices_low] = -1000.0
    contrast_data[indices_high] = 2000.0

    label_data = np.zeros(shape)

    config = StatisticsOptions(range_brackets=[1.0, 1.0, 1.0, 1.0])
    config.parameter = 0.1
    winsoring_result = generate_tissue_types_from_sample(contrast_data, label_data, 0, config)

    config.method = "median"
    config.parameter = 0.3
    median_result = generate_tissue_types_from_sample(contrast_data, label_data, 0, config)

    config.method = "gaussian"
    gaussian_result = generate_tissue_types_from_sample(contrast_data, label_data, 0, config)

    # Check if the estimated mean and standard deviation is within reasonable bounds

    assert abs(winsoring_result.mean_range[0] - mean) < mean * 0.01
    assert abs(winsoring_result.stddev_range[0] - std_dev) < std_dev * 0.05
    assert abs(median_result.mean_range[0] - mean) < mean * 0.01
    assert abs(median_result.stddev_range[0] - std_dev) < std_dev * 0.2

    # For the Gaussian method, we expect much higher deviations
    # We just check that the error is indeed > 30% for the mean and > 100% for the standard deviation
    assert abs(gaussian_result.mean_range[0] - mean) > mean * 0.3
    assert abs(gaussian_result.stddev_range[0] - std_dev) > std_dev * 1.0

