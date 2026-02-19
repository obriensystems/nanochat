#!/bin/bash

#- andrej.karpathy Nanochat on NVIDIA DGX Spark GB10 - https://github.com/ObrienlabsDev/blog/issues/170
#- https://github.com/obriensystems/nanochat/blob/dev/runs/runcpu.sh#L79
#- DGX Spark https://github.com/ObrienlabsDev/foundation-transformer-llm/issues/7
#- Apple Silicon https://github.com/ObrienlabsDev/foundation-transformer-llm/issues/8
#- Windows/WSL NVIDIA https://github.com/ObrienlabsDev/foundation-transformer-llm/issues/9
#- https://github.com/karpathy/nanochat/issues/542

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


