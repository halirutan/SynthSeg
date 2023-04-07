import numpy as np
from pathlib import Path
import pickle


def test_build_model_inputs(model_inputs):
    # TODO: how should we deal with large files?
    model_inputs_path = Path(__file__).parent / "model_inputs_seed43.pkl"

    with model_inputs_path.open(mode="br") as f:
        # pickle.dump(model_inputs, f)
        model_inputs_target = pickle.load(f)

    for inp, tar in zip(model_inputs, model_inputs_target):
        assert np.allclose(inp, tar)
