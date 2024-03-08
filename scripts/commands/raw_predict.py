import numpy as np
from typing import List
from SynthSeg.training_options import TrainingOptions

def adjust_size(input_data: np.ndarray, desired_shape: List[int]):
    current_shape = input_data.shape
    adjustments = [(ds - cs) / 2. for ds, cs in zip(desired_shape, current_shape)]
    pad_values = [[int(np.floor(adj)), int(np.ceil(adj))] if adj > 0 else [0, 0] for adj in adjustments]
    crop_values = [[-int(np.floor(-adj)), int(np.ceil(-adj))] if adj < 0 else [0, 0] for adj in adjustments]
    content_values = [[int(np.floor(adj)), int(cs + np.ceil(adj))] for adj, cs in zip(adjustments, current_shape)]
    # Pad / Crop the array for non-negative / negative adjustments (respectively)
    input_data_adjusted = np.pad(input_data, pad_width=pad_values, mode='constant', constant_values=0)
    input_data_adjusted = input_data_adjusted[
                          crop_values[0][0]:content_values[0][1], crop_values[1][0]:content_values[1][1],
                          crop_values[2][0]:content_values[2][1]]

    return input_data_adjusted


def predict(train_opts: TrainingOptions):
    pass