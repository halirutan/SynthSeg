# Python Environment using Miniforge and Mamba

Miniforge is a Python package manager that includes conda and mamba.
Mamba is basically conda, only faster, especially during the resolution of dependencies.
Most conda commands can also be used in Mamba, but Mamba has additional nice features like
`mamba repoquery search`.

Here, we use Mamba to install the environment and resolve all package dependencies.
After that, it is fine to use `conda` everywhere else.
To install Miniforge, create the environment or update the environment you can use

```shell
bash bootstrap_mamba.sh
```
or

```shell
bash bootstrap_mamba.sh /path/to/existing/miniforge
```
Depending on if you're operating system is Linux or macOS, this script will use
the `environment.yml` or `environment_osx.yml` file as the definition for the Python environment.
Other operating systems are currently not supported.

The script will perform the following steps:

1. Check if Miniforge is installed either in `$HOME/miniforge` or the path provided to the script.
2. If not, it will download and install Miniforge.
3. Check if Miniforge has already an environment with the same name as specified in `environment.yml`
   - If yes, it will call `mamba env update -f "environment.yml"` which checks if any of the package versions in the file have changed and update the environment accordingly.
   - If no, it will call `mamba env create -f "environment.yml"` to create a new environment.

## Install SynthSeg Locally

After creating the Mamba environment, you need to activate it using

```shell
conda activate synth_seg_py10
```

Then you can make the source code package available to Mamba by using
the following in the root directory of the project:

```shell
pip install -E .
```

Now, all command line tools will be available directly in the terminal
without having to call python.
All scripts are prefixed with `synthSeg-`.
Changes in the source code will be immediately available without requiring to
re-install the package.

## Test Job for Raven

The `raven_test_job` directory contains a small example that uses tensorflow to train a network. It can be used
to check if the Python environment indeed works and makes use of the GPU.
You can submit this job on the Raven cluster using

```bash
sbatch test_job.sh
```

and check its status using `squery -u yourLogin`.
Once it is finished, you will find the log-files of the run inside the directory.

Note that this test-job currently assumes that the mamba environment is called `synth_seg_py10` and that you
indeed have installed mamba in `$HOME/miniforge`.
Additionally, the line

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT}"
```

inside `test_job.sh` ensures that tensorflow can find CUDA that was loaded with `module load cuda/11.6`.

Bootstrapping Miniforge, creating a fresh environment, and running the job doesn't take more than 10min.