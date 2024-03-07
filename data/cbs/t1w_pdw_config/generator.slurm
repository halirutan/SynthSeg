#!/bin/bash -l
# For many training sets, this script is intended to be called with, e.g.
# sbatch --array=0-19 generator.slurm
#
# This will create datasets in parallel and the only thing you have to take care of
# is that the filenames from different workers don't match. This is taken care of
# by including the SLURM_ARRAY_TASK_ID in the generation script.

# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J synthseg
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=pscheibe@cbs.mpg.de
#SBATCH --time=24:00:00


output_dir=/u/pscheibe/ptmp_link/t1w_pdw_500mu_training_tfrecords
config_file=${output_dir}/generator.yml

module load anaconda/3/2021.11 scikit-learn/1.1.1 tensorflow/gpu-cuda-11.6/2.11.0 keras/2.11.0 tensorboard/2.11.0
source /u/pscheibe/mambaforge/etc/profile.d/conda.sh
conda activate synth_seg_py39

# --- TF related flags (provided by Tim) ---

# Avoid CUPTI warning message
LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64/:${LD_LIBRARY_PATH}

# Avoid OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

## XLA
# cuda aware
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
# enable autoclustering for CPU and GPU
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# ---
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 2 > nvidia_smi_monitoring.csv &
NVIDIASMI_PID=$!

# Start it with: sbatch --array=0-19 generator.slurm
# With count=2000 and batchsize in the generator.yml set to 8, this will result in the following number of training sets
# workers * count * 8 = 20 * 2000 * 8 = 320000
srun synthSeg-generate-synth-images \
--output_dir=${output_dir} \
--config_file=${config_file} \
--count=2000 \
--start_int=$(($SLURM_ARRAY_TASK_ID*2000)) \
2>&1 | tee -a generation${SLURM_ARRAY_TASK_ID}.log

kill $NVIDIASMI_PID
