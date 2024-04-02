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
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=13:00:00


code_dir=/u/dcfidalgo/projects/cbs/segmentation/SynthSeg
output_dir=/ptmp/dcfidalgo/projects/cbs/segmentation/generation/v2
count=500

module purge
module load apptainer/1.2.2


srun apptainer exec \
	--nv \
	--bind /ptmp/dcfidalgo \
	--env PYTHONPATH=/u/dcfidalgo/projects/cbs/segmentation/SynthSeg,TF_FORCE_GPU_ALLOW_GROWTH=false \
	/u/dcfidalgo/projects/cbs/segmentation/SynthSeg/container/nvidia_tensorflow.sif python /u/dcfidalgo/projects/cbs/segmentation/SynthSeg/scripts/slurm/generation.py \
		--output_dir=${output_dir} \
		--config_file=${output_dir}/config.yml \
		--tfrecord \
		--count=${count} \
		--start_int=$(($SLURM_ARRAY_TASK_ID*${count})) \
		2>&1 | tee -a generation${SLURM_ARRAY_TASK_ID}.log

