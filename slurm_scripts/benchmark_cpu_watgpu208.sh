#!/bin/bash
#SBATCH --mem=32G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=2
#SBATCH --time=720
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --mincpus=16
#SBATCH --nodelist=watgpu208
conda run -n vllm_throughput_cpu python run.py \
    --num-warmup-runs 1 \
    --num-runs 3 \
    --models Qwen/Qwen3-0.6B JackFram/llama-68m \
    --num-concurrent-requests 1 2 4 8 16 32 64 128 \
    --num-input-tokens 1 2 4 8 16 32 64 128 256 512 \
    --num-output-tokens 1 2 4 8 16 \
    --cpu-omp-threads-binds 0-7 0-15 0-5 0-5,8-13
