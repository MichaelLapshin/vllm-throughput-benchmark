#!/bin/bash
#SBATCH --output=slurm_watgpu108_cpu_output.out
#SBATCH --mem=100G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=2
#SBATCH --time=720
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=64
#SBATCH --mincpus=64
#SBATCH --nodelist=watgpu108
newgrp perf_users
conda activate vllm_throughput_cpu
python run.py \
    --num-warmup-runs 1 \
    --num-runs 3 \
    --models Qwen/Qwen3-8B \
    --num-concurrent-requests 1 \
    --num-input-tokens 1 \
    --num-output-tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32767 \
    --cpu-omp-threads-binds None

# This benchmark is for verifying the memory-boundedness of decoding.
# We expect to find that the ratio between GPU and CPU throughput stays inline with the ratio in memory bandwidth.
#   CPU: 
#       gpu01: 2133MHz x 4 channels x 64 bits / 8 bits per byte = 68.256 GBps
#       watgpu108: 4800 x 8 channels x 64 / 8 bits per byte = 307.2 GBps
#   GPU:
#       RTX 6000 ADA: 960 GBps
#       H200 NVL: 4,800 GBps
