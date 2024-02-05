#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_logs/job_4gpu.out.%j
#SBATCH -e ./job_logs/job_4gpu.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J synthseg
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#
#SBATCH --mail-type=end
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH --time=00:15:00



cfg_name="config_gpu4_weak_test.yml"


echo "Use containers"
source /etc/profile.d/modules.sh
module purge
module load apptainer/1.2.2

cfg_file="$HOME/Projects/SynthSeg/project_folder/training_configs/test/${cfg_name}"
config_path_original="/ptmp/nhorlava/projects/SynthSeg/training_configs/test"
data_path_original="/ptmp/dcfidalgo/projects/cbs/segmentation/generation/v1/tfrecords"
code_path="$HOME/Projects/SynthSeg"
container_path=$HOME/Projects/SynthSeg/container/nvidia_tensorflow.sif
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,pci.bus_id --format=csv -l 2 > "nvidia_smi/nvidia_smi_monitoring_4gpu.csv" &
NVIDIASMI_PID=$!

srun apptainer exec \
        --nv \
        -B $config_path_original:$config_path_original,$data_path_original:$data_path_original \
        --env PYTHONPATH=$code_path \
        $container_path python $HOME/Projects/SynthSeg/scripts/slurm/training.py --cfg_file=$cfg_file

kill $NVIDIASMI_PID
