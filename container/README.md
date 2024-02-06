# Getting started 

The container was build on tensorflow version==2.14 with together with additional requirements as scpeficied in `containers/requirement_container.txt`. Once you will need to install another package or its version, modify `containers/requirement_container.txt` and [re-build](https://gitkraken.dev/link/dnNjb2RlOi8vZWFtb2Rpby5naXRsZW5zL2xpbmsvci81ZWJhYjFlNjRlMDI1NmU4ZjVhM2M3ZDJjOWVjYmRjZjQzYmVkNjc5L2YvY29udGFpbmVyL1JFQURNRS5tZD91cmw9Z2l0JTQwZ2l0aHViLmNvbSUzQWhhbGlydXRhbiUyRlN5bnRoU2VnLmdpdCZsaW5lcz03?origin=gitlens) the container.

To run a container, you could either re-build it from the recipy yourself or, in case of working on RAVEN cluster, copy the working image to you home directory or simply link it via symbolic link. 

1. To re-build a conntainer from scratch on RAVEN, and assuming you'rein the root directory of the project, run:

```shell
module load apptainer/1.2.2
apptainer build --fakeroot container/nvidia_tensorflow.sif container/nvidia_tensorflow.def
``` 

2. If you're working on RAVEN, you could access already created image of a container, which is located in the `/ptmp/nhorlava/projects/SynthSeg/container/nvidia_tensorflow.sif`. 
You could also link it to your working directory via symbolic link: 

```shell
 ln -s /ptmp/nhorlava/projects/SynthSeg/container/nvidia_tensorflow.sif $PWD/container/nvidia_tensorflow_linked.sif
```

# Running a container

An example of sbatch file for training SynthSeg from scratch is located in `scripts/slurm/container_example/training_gpu4.slurm`. An example of resuming the SynthSeg training is located in `scripts/slurm/container_example/training_gpu4_resume.slurm`
You'll need to modify it according to location of your code / data/ models. 

Let's dive into example or running a container:


```bash=
#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_logs/job_4gpu_wo_opt.out.%j
#SBATCH -e ./job_logs/job_4gpu_wo_opt.err.%j
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
#SBATCH --mail-user=username@mpcdf.mpg.de
#SBATCH --time=00:05:00


source /etc/profile.d/modules.sh
module purge
module load apptainer/1.2.2

cfg_name="config_gpu4.yml"
config_path_original="/ptmp/nhorlava/projects/SynthSeg/training_configs/test"
cfg_file="${config_path_original}/${cfg_name}"
data_path_original="/ptmp/dcfidalgo/projects/cbs/segmentation/generation/v1/tfrecords"
code_path="$HOME/Projects/SynthSeg"
container_path=$HOME/Projects/SynthSeg/container/nvidia_tensorflow.sif


srun apptainer exec \
        --nv \
        -B $config_path_original:$config_path_original,$data_path_original:$data_path_original \
        --env PYTHONPATH=$code_path \
        $container_path python $HOME/Projects/SynthSeg/scripts/slurm/training.py --cfg_file=$cfg_file

```

1. **Binding external storage**: 

    - My configs, as well as the data I'm using, are located on **/ptmp** filesystem. Withing a container instance, this filesystem will not be visible unless its binded. 
    - Hence, we provide the path to where configs/models are stored 
([`config_path_original`](https://gitkraken.dev/link/dnNjb2RlOi8vZWFtb2Rpby5naXRsZW5zL2xpbmsvci81ZWJhYjFlNjRlMDI1NmU4ZjVhM2M3ZDJjOWVjYmRjZjQzYmVkNjc5L2YvY29udGFpbmVyL1JFQURNRS5tZD91cmw9Z2l0JTQwZ2l0aHViLmNvbSUzQWhhbGlydXRhbiUyRlN5bnRoU2VnLmdpdCZsaW5lcz01Nw%3D%3D?origin=gitlens)),
 as well as were the data is stored
([`data_path_original`](https://gitkraken.dev/link/dnNjb2RlOi8vZWFtb2Rpby5naXRsZW5zL2xpbmsvci81ZWJhYjFlNjRlMDI1NmU4ZjVhM2M3ZDJjOWVjYmRjZjQzYmVkNjc5L2YvY29udGFpbmVyL1JFQURNRS5tZD91cmw9Z2l0JTQwZ2l0aHViLmNvbSUzQWhhbGlydXRhbiUyRlN5bnRoU2VnLmdpdCZsaW5lcz01OQ%3D%3D?origin=gitlens)). 

    - When running a python code withing a container instance, we first bind this directories via [**`-B`** ](https://gitkraken.dev/link/dnNjb2RlOi8vZWFtb2Rpby5naXRsZW5zL2xpbmsvci81ZWJhYjFlNjRlMDI1NmU4ZjVhM2M3ZDJjOWVjYmRjZjQzYmVkNjc5L2YvY29udGFpbmVyL1JFQURNRS5tZD91cmw9Z2l0JTQwZ2l0aHViLmNvbSUzQWhhbGlydXRhbiUyRlN5bnRoU2VnLmdpdCZsaW5lcz02Ng%3D%3D?origin=gitlens) flag .

2. **Providing path to developed package**:
    - Assume the code is still in the development ohase and therefore is still being modified. At the same time, we want to treat it as a panckage. Now, if we were to run **without containers**, we could have just instaled it in a virtual environment via `pip install -e .`. This would have installed our package in a "editable" mode from your project directory, meaning that you could make changes to the source code without needing to reinstall the package every time you make a change.
    - To retain this behavior when running a container, we will instead pass the `$PYTHONPATH` variable to the container instance. By doing so, we will modify python's module search path, therefore making python look into the specified directorie(s) when trying to import a module. 
    - To pass the '$PYTHONPATH' to the container instance when running it, we'll use the [**`--env`** ](https://gitkraken.dev/link/dnNjb2RlOi8vZWFtb2Rpby5naXRsZW5zL2xpbmsvci81ZWJhYjFlNjRlMDI1NmU4ZjVhM2M3ZDJjOWVjYmRjZjQzYmVkNjc5L2YvY29udGFpbmVyL1JFQURNRS5tZD91cmw9Z2l0JTQwZ2l0aHViLmNvbSUzQWhhbGlydXRhbiUyRlN5bnRoU2VnLmdpdCZsaW5lcz02Nw%3D%3D?origin=gitlens) flag. 
    - **NOTE**: Make sure to change **code_path** to the directoty of **YOUR** project.

# Notes on symbolic links in a bundle with containers

ALthough it might be common to use symbolic links, e.g. for folders located on aother filesystem, the container instance will not see them. 

Therefore, provide the full original path for the container, as well as bind it if it is located on another filesystem. 

Imaging you have created a symbolic link to your config folder: 

```
ln -s /ptmp/dcfidalgo/projects/cbs/segmentation/generation/v1/tfrecords/ project_folder/training_containers
```

Now, you will see in you project directory smth like: 

```
|_SynthSeg
    |___ SynthSeg
    |___ container
    |___ ...
    |___ project_folder
        |___ training_containers
            |___ config_gpu4.yml
    |___ ...

```

To now run a container while providing path to `config_gpu4.yml`, you'd need to bind teh **original** path of this file: 

```shell
cfg_path="$(readlink project_folder/training_containers)"
```

And then bind this original path when running a container instance to the exact path: 

```shell

apptainer shell \
    -B $cfg_path:$cfg_path \
     container/nvidia_tensorflow.sif python yourpythoscript.py
```

 NOTE: 
 - It is adviced agaist binding this remote path to the same directory to which your symbolic link is connected, like `apptainer shell -B $cfg_path:$PWD/project_folder/training_containers container/nvidia_tensorflow.sif` as it will not see the content of `/project_folder/training_containers` in this case:

 ```console
Apptainer> ls project_folder/training_configs
project_folder/training_configs
 ```
 -  Similarly, if you would bind the remote path to the new folder on a host file system, an empty folder will be created, while files created will be stored in the original remote directory:
 `apptainer shell -B $cfg_path:$PWD/project_folder/training_containers container/nvidia_tensorflow.sif` as it will not see the content of `/project_folder/training_containers` in this case, and will create files in original remote directory, as in mocking example:

 ```console
user@raven02> cfg_path="/ptmp/nhorlava/projects/SynthSeg/test_sl" 
user@raven02> apptainer shell -B $cfg_path:$PWD/project_folder/test_sl_linked container/nvidia_tensorflow.sif
INFO:    fuse2fs not found, will not be able to mount EXT3 filesystems
INFO:    gocryptfs not found, will not be able to use gocryptfs
13:4: not a valid test operator: (
13:4: not a valid test operator: 
Apptainer> touch project_folder/test_sl_linked/test.txt
Apptainer> ls project_folder/test_sl_linked
test.txt
Apptainer> exit

user@raven02> ls project_folder/test_sl_linked/
user@raven02> ls $config_path
user@raven02> ls $cfg_path
test.txt
 ```
  


