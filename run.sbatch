#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu-2080ti
#SBATCH --time=1-00:00
#SBATCH --no-requeue

#SBATCH --job-name=COOD
#SBATCH --error=/mnt/qb/work/bethge/twiedemer43/logs/sbatch/%x_%j.err
#SBATCH --output=/mnt/qb/work/bethge/twiedemer43/logs/sbatch/%x_%j.out

CFG="cfg.yml"

set -e # exit when any command fails

cd /mnt/qb/work/bethge/twiedemer43/code/cood
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn \
  singularity exec --nv \
    -B /scratch_local \
    -B /mnt/qb/work/bethge/ \
    /mnt/qb/work/bethge/twiedemer43/container/cood.sif \
      python -u main.py -c $CFG