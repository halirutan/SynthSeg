#!/bin/bash -l
# Standard output and error:
#SBATCH -o job.out.%j
#SBATCH -e job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J synthseg
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
##SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=24:00:00


code_dir=/u/dcfidalgo/projects/cbs/segmentation/SynthSeg
output_dir=/ptmp/dcfidalgo/projects/cbs/segmentation/generation/v2
skip=0

module purge
module load apptainer/1.2.2


srun apptainer exec \
	--nv \
	--bind /ptmp/dcfidalgo \
	--env PYTHONPATH=${code_dir}:$PYTHONPATH,TF_FORCE_GPU_ALLOW_GROWTH=false \
	${code_dir}/container/nvidia_tensorflow.sif python ${code_dir}/scripts/slurm/generation/tfrecord_to_nifti.py \
		--input=${output_dir}/tfrecords \
		--output=${output_dir}/niftis \
		--config=${output_dir}/config.yml \
		--skip=$skip \
		--tar \
		2>&1 | tee -a tfrecords_to_niftis.log

