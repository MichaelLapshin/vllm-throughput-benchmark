#!/bin/bash
#SBATCH --output=slurm_gpu_watgpu508.out
#SBATCH --mem=32G
#SBATCH --sockets-per-node=1
#SBATCH --threads-per-core=2
#SBATCH --time=960
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --mincpus=16
#SBATCH --nodelist=watgpu508
conda run -n vllm_throughput_gpu python run.py \
    --num-warmup-runs 1 \
    --num-runs 5 \
    --models JackFram/llama-68m JackFram/llama-160m huggyllama/llama-7b huggyllama/llama-13b huggyllama/llama-65b \
    --num-concurrent-requests 1 2 4 8 16 32 64 128 \
    --num-input-tokens 1 2 4 8 16 32 64 128 256 \
    --num-output-tokens 1 2 4 8 16 32 64 128 256
