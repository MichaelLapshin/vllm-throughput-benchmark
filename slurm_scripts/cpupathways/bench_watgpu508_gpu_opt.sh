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
    --models facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/opt-30b facebook/opt-66b \
    --num-concurrent-requests 1 2 4 8 16 32 64 128 \
    --num-input-tokens 1 2 4 8 16 32 64 128 256 \
    --num-output-tokens 1 2 4 8 16 32 64 128 256
