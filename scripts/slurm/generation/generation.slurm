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
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=13:00:00


code_dir=/u/dcfidalgo/projects/cbs/segmentation/SynthSeg
output_dir=/ptmp/dcfidalgo/projects/cbs/segmentation/generation/v3
count=250

module purge
module load apptainer/1.2.2


srun apptainer exec \
	--nv \
	--bind /ptmp/dcfidalgo \
	--env PYTHONPATH=${code_dir},TF_FORCE_GPU_ALLOW_GROWTH=false \
	${code_dir}/container/nvidia_tensorflow.sif python ${code_dir}/scripts/slurm/generation.py \
		--output_dir=${output_dir} \
		--config_file=${output_dir}/config.yml \
		--count=${count} \
		--start_int=$(($SLURM_ARRAY_TASK_ID*${count})) \
		2>&1 | tee -a generation${SLURM_ARRAY_TASK_ID}.log

