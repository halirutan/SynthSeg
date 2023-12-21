#!/bin/bash

# Load necessary modules
module purge
module load anaconda/3/2021.11 scikit-learn/1.1.1 tensorflow/gpu-cuda-11.6/2.11.0 keras/2.11.0 tensorboard/2.11.0

# If venv deoes not exist create it
if [ ! -f ${HOME}/venvs/synthseg/bin/activate ]
then
    echo "venv does not exist - Creating it..."
    mkdir -p ${HOME}/venvs/synthseg
    python -m venv --system-site-packages ${HOME}/venvs/synthseg
    source ${HOME}/venvs/synthseg/bin/activate
    pip install -r requirements_raven.txt
    pip install -e .
else
    # Activate venv
    source ${HOME}/venvs/synthseg/bin/activate
fi


