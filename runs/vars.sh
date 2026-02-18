#!/bin/bash

# michael at obrienlabs.dev modifications
# for NVIDIA only
#export CPU_GPU=gpu
# for CPU and MPS (Apple GPU)
export CPU_GPU=cpu

# change these
export DEPTH=5
# note Tokens / micro-batch: must be = or < total batch of 16k
export BATCH_SIZE=32

# keep these defaulted usually
export ITERATIONS_BASE_TRAIN=1000
export ITERATIONS_SFT=1500
export TOTAL_BATCH_SIZE=16384

# these are derived
export MODEL_TAG="d${DEPTH}"

## paralleization for cpu only
#export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
#export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)


