#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# This script was last updated/tuned on Jan 17, 2026.

# Run as:
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# all the setup stuff
# michael at obrienlabs.dev modifications
# for NVIDIA only
#export CPU_GPU=gpu
# for CPU and MPS (Apple GPU)
export CPU_GPU=cpu
export DEPTH=5
# note Tokens / micro-batch: must be = or < total batch of 16k
export BATCH_SIZE=32
export MODEL_TAG="d${DEPTH}"
export ITERATIONS=1000

## paralleization
#export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
#export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo $timestamp
START_TIME=$(date +%s)

rm -rf ~/.cache/nanochat/chatsft_checkpoints/* -f
rm -rf ~/.cache/nanochat/base_checkpoints/* -f
rm -rf ~/.cache/nanochat/base_eval/* -f

echo "Running: $CPU_GPU: depth:$DEPTH batch:$BATCH_SIZE model:$MODEL_TAG"
python -m nanochat.report reset

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra $CPU_GPU
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

echo "base_train"
# 4 batch for d26
python -m scripts.base_train \
    --depth=$DEPTH \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=$BATCH_SIZE \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=$ITERATIONS \
    --run=$WANDB_RUN
    #    --fp8 \
    #--target-param-data-ratio=8.25 \

echo "base_eval"
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16
#python -m scripts.chat_eval -i sft -g $MODEL_TAG

# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
echo "chat_sft"
python -m scripts.chat_sft \
    --device-batch-size=$BATCH_SIZE \
    --eval-every=200 \
    --model-tag=$MODEL_TAG \
    --num-iterations=1500 \
    --run=$WANDB_RUN \
    --max-seq-len=512 \
    --eval-tokens=524288 
    --total-batch-size=16384

# python -m scripts.chat_eval -i sft -g $MODEL_TAG

python -m nanochat.report generate
# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
python -m scripts.chat_cli -p "What is the capital of France?" --model-tag $MODEL_TAG

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo $timestamp
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "duration, $ELAPSED_TIME, depth, $DEPTH, batch, $BATCH_SIZE, model, $MODEL_TAG"