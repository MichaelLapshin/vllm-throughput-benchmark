#!/bin/bash
#SBATCH --output=slurm_gpu_watgpu708.out
#SBATCH --mem=32G
#SBATCH --sockets-per-node=1
#SBATCH --threads-per-core=2
#SBATCH --time=720
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=24
#SBATCH --mincpus=24
#SBATCH --nodelist=watgpu708
conda run -n vllm_throughput_gpu python run.py \
    --num-warmup-runs 1 \
    --num-runs 2 \
    --models JackFram/llama-68m Qwen/Qwen3-0.6B deepseek-ai/deepseek-coder-1.3b-instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B \
    --num-concurrent-requests 1 2 4 8 16 32 64 128 \
    --num-input-tokens 1 2 4 8 16 32 64 128 256 \
    --num-output-tokens 1 2 4 8 16 32 64 128 256
