#!/bin/bash
#SBATCH --mem=32G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH --time=720
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=64
#SBATCH --mincpus=64
#SBATCH --nodelist=watgpu608
conda run -n vllm_throughput_cpu python run.py \
    --num-warmup-runs 1 \
    --num-runs 3 \
    --models Qwen/Qwen3-0.6B JackFram/llama-68m \
    --num-concurrent-requests 1 2 4 8 16 32 64 128 \
    --num-input-tokens 1 2 4 8 16 32 64 128 256 512 \
    --num-output-tokens 1 2 4 8 16 \
    --cpu-omp-threads-binds 0-31 0-63 0-15 0-15,32-47
