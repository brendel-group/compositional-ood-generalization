#!/bin/bash
#SBATCH --job-name=grid_search    # Job name
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-2080ti
#SBATCH --time=24:00:00
#SBATCH --mem=20G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/qb/work/bethge/twiedemer43/sscript/logs/sbatch/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/twiedemer43/sscript/logs/sbatch/%x_%j.err   # File to which STDERR will be written
USER=twiedemer43

# necessary during conversion of docker container to find required packages
export PATH=$PATH:/usr/sbin

# log infos
scontrol show job=$SLURM_JOB_ID
echo $SLURM_JOB_ID
nvidia-smi

# build and run the singularity container
srun singularity exec --nv \
    --bind /mnt/qb/work/bethge/$USER \
    docker://pytorch/pytorch \
    bash -c "cd /mnt/qb/work/bethge/twiedemer43/code/identify_ood; ~/.conda/envs/iood/bin/python3.9 train.py"