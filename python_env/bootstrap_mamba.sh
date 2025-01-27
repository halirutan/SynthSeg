#!/bin/bash

# This is to prevent conda/mamba to run into situations
# where it can't solve the environment because TF has a weird
# way of specifying it's CUDA dependency.
export CONDA_OVERRIDE_CUDA="11.8"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

if [ -z "$1" ]; then
  MINIFORGE_PATH="${HOME}/miniforge"
else
  MINIFORGE_PATH="$1"
fi
CONDA_INIT="${MINIFORGE_PATH}/etc/profile.d/conda.sh"

echo "CONDA_INIT ${CONDA_INIT}"

# First check valid OS and if the environment.yml file does exist in the script directory
OS=$(uname)
if [ "$OS" == "Linux" ]; then
  ENV_FILE="${SCRIPT_DIR}/environment.yml"
elif [ "$OS" == "Darwin" ]; then
  ENV_FILE="${SCRIPT_DIR}/environment_osx.yml"
else
  echo "Unsupported operating system: $OS. This script only works on Linux or OSX."
  exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
  echo "Could not find $ENV_FILE file. This isn't going to work out between us my friend."
  exit 1
fi

# Extract the name of the environment
ENV_NAME=$(sed -n '1s/name: //p' "${ENV_FILE}")

# Check if we already have mamba. If not, we install it into $HOME/mambaforge
if [ -f "$CONDA_INIT" ]; then
  echo "Miniforge seems to be already installed."
else
  echo "Couldn't find miniforge. Installing it now..."
  wget -O Miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge.sh -b -p "${MINIFORGE_PATH}"
fi

# Now the conda.sh and mamba.sh must exist or we're screwed
if [ -f "$CONDA_INIT" ]; then
  echo "Initializing Conda"
  source "${CONDA_INIT}"
else
  echo "Something went wrong during the installation. Can't find ${CONDA_INIT}"
  exit 1
fi

# Check if the conda environment already exist. If yes, we update it. If not, we create it from the environment.yml
if mamba env list | grep -q "^${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists. Running an update according to ${ENV_FILE}"
  mamba env update -f "${SCRIPT_DIR}/environment.yml"
else
  echo "Environment ${ENV_NAME} does not exist. Creating it according to ${ENV_FILE}"
  mamba env create -f "${SCRIPT_DIR}/environment.yml"
fi
