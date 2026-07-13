#!/bin/bash
#SBATCH --output=slurm_watgpu108_cpu_32cores.out
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
    --num-concurrent-requests 1 2 4 8 16 32 64 128 256 512 \
    --num-input-tokens 1 \
    --num-output-tokens 1 \
    --cpu-omp-threads-binds None

# This benchmark is for verifying that the CPU is more compute-bound than the GPU during the decode-phase.
#   CPU: Should be compute-bound.
#        An additional FFN is needed for each request.
#        The attention-matrix is small, so that would only scale constantly.
#   GPU: Should be able to scale much more.
#        ~13x more on RTX ADA 6000 compared to Intel Xeon Gold 6448H.
#        ~100x more on RTX ADA 6000 compared to Intel Xeon E5-2640v3.
