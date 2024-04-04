# Getting started 

The container recipy (located in `container/nvidia_tensorflow.def`) was modified to include additional mlflow package. Please re-build the container before proceeding.

To re-build a container from scratch on RAVEN, and assuming you're in the root directory of the project, run:

```shell
module load apptainer/1.2.2
apptainer build --fakeroot container/nvidia_tensorflow.sif container/nvidia_tensorflow.def
``` 

# Running the training with mlflow tracking 

An example of sbatch file for training SynthSeg with mlflow tracking is located in `scripts/slurm/mlflow_example/training.slurm`. 
Let's dive into an example:


```shell
#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_logs/job.out.%j
#SBATCH -e ./job_logs/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J synthseg
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256GB
#
## SBATCH --partition=gpudev
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#
#SBATCH --mail-type=end
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH --time=00:15:00

cfg_file="/ptmp/nhorlava/projects/SynthSeg/training_configs/test_mlflow/config.yml"
sif_file=$HOME/Projects/SynthSeg/container/nvidia_tensorflow.sif

code_dir="$HOME/Projects/SynthSeg"
data_dir=/ptmp/dcfidalgo/projects/cbs/segmentation/generation/v2
output_dir=/ptmp/nhorlava/projects/SynthSeg/training_configs

module purge
module load apptainer/1.2.2

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 2 > nvidia_smi_monitoring.csv &
NVIDIASMI_PID=$!

apptainer_opts=(
  --nv
  -B $code_dir,$data_dir,$output_dir
  --env PYTHONPATH=$code_dir
  # in the container this is set to true by default
  --env TF_FORCE_GPU_ALLOW_GROWTH=true
  --env MLFLOW_CONFIG="$HOME/Projects/SynthSeg/scripts/slurm/mlflow_example/config.ini"
  
)
	
srun apptainer exec \
  "${apptainer_opts[@]}" \
  $sif_file python $code_dir/scripts/slurm/training.py --cfg_file=$cfg_file

kill $NVIDIASMI_PID
```

1. **Modification of configuration file (`cfg_file`)**: 

Similar to using WB, if you want to use mlflow tracking, you would need to specify so in the config file: 

```shell
mlflow: True
mlflow_log_freq: epoch
```

The behavior of `mlflow_log_freq` argument is the same as the behavior of `wandb_log_freq` argument: 

- if "epoch", logs metrics at the end of each epoch.
- If "batch", logs metrics at the end of each batch.
- If an integer, logs metrics at the end of that many batches.


2. **Providing a file with mlflow specs**: 

To run Mlflow tracking, one needs to provide a configuration file with mlflow-specific arguments. An example of such config can be found in `scripts/slurm/mlflow_example/config.ini` or below: 

```
[LOGIN]
USERNAME = your_username
PASSWORD =  your_password

[HPC_CLOUD]
URI = http://mlflowinstance.nhorlava-dev.hpccloud.mpg.de:5000

[MLFLOW]
EXPERIMENT = synthseg
```

Before submitting the job, please replace USERNAME and PASSWORD with your username/password. 





