#!/bin/bash
#SBATCH --output=slurm_watgpu108_cpu_input.out
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
    --num-input-tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32767 \
    --num-output-tokens 1 \
    --cpu-omp-threads-binds None

# This benchmark is for verifying that CPU is compute-bound for the prefill phase. While the GPU is less-so.
#   CPU: matter the number of tokens, the compute time should be proportional.
#   GPU: should be able to scale pretty well. Up to a certain point.
# 
# Bonus expectiation:
#   On RTX ADA 6000, we should hit a memory-wall that will either start swapping or OOM the program.
#   So we would expact a sudden spike in latency to compute 65536 input tokens. 
