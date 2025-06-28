#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=rtx_4090:1
#SBATCH --gres=gpumem:20g
#SBATCH --time=01:00:00
#SBATCH --job-name=jax-gpu
#SBATCH --output=/cluster/home/mpundir/dev/spectralsolvers/benchmarking/analysis-gpu.out
#SBATCH --error=/cluster/home/mpundir/dev/spectralsolvers/benchmarking/analysis-gpu.err

source /cluster/project/cmbm/local-stacks/load-scripts/load_gpu.sh 
source /cluster/work/cmbm/mpundir/venv/my-venv/bin/activate
export JAX_CACHE_DIR="$SCRATCH/jax-cache-gpu"
export JAX_PLATFORM="gpu"
export SPECTRAL_LIB_PATH="/cluster/home/mpundir/dev/spectralsolvers/"

python $SPECTRAL_LIB_PATH/benchmarking/benchmark_linear_elasticity.py
