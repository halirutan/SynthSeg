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
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=24:00:00

cfg_file=/ptmp/dcfidalgo/projects/cbs/segmentation/training/v2/config.yml
sif_file=/u/dcfidalgo/projects/cbs/segmentation/SynthSeg/container/nvidia_tensorflow.sif

code_dir=/u/dcfidalgo/projects/cbs/segmentation/SynthSeg
data_dir=/ptmp/dcfidalgo/projects/cbs/segmentation/generation/v2
output_dir=/ptmp/dcfidalgo/projects/cbs/segmentation/training/v2


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
  # enable autoclustering for CPU and GPU
  #--env TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
  # Useful for debugging
  #--env NCCL_DEBUG=INFO
  --env WANDB_BASE_URL="http://10.186.1.194:80"
  --env WANDB_API_KEY="local-3277de3bee22b1c6c3e074cfc1d73b7b169f5d41"
  --env WANDB_PROJECT="synthseg"
  --env WANDB_NAME="v2"
)
	
srun apptainer exec \
  "${apptainer_opts[@]}" \
  $sif_file python $code_dir/scripts/slurm/training.py --cfg_file=$cfg_file

kill $NVIDIASMI_PID
