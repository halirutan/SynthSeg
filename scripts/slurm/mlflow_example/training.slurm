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
