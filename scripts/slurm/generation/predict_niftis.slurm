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
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=24:00:00

module purge
module load apptainer/1.2.2

srun apptainer run \
	--nv \
	--bind /ptmp/dcfidalgo \
	/u/dcfidalgo/projects/cbs/segmentation/SynthSeg/container/synthseg_predict.sif \
		--i /ptmp/dcfidalgo/projects/cbs/segmentation/generation/v2/niftis/images2 \
		--o /ptmp/dcfidalgo/projects/cbs/segmentation/generation/v2/niftis/imasks \
		--robust

