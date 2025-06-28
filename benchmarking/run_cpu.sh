#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8096
#SBATCH --time=01:00:00
#SBATCH --job-name=jax-cpu
#SBATCH --output=/cluster/home/mpundir/dev/spectralsolvers/benchmarking/analysis-cpu.out
#SBATCH --error=/cluster/home/mpundir/dev/spectralsolvers/benchmarking/analysis-cpu.err

source /cluster/project/cmbm/local-stacks/load-scripts/load_fenicsx.sh 
source /cluster/work/cmbm/mpundir/venv/my-venv/bin/activate
export JAX_CACHE_DIR="$SCRATCH/jax-cache"
export JAX_PLATFORM="cpu"
export SPECTRAL_LIB_PATH="/cluster/home/mpundir/dev/spectralsolvers/"

python $SPECTRAL_LIB_PATH/benchmarking/benchmark_linear_elasticity.py