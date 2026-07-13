#!/bin/bash
#SBATCH --output=slurm_cpu_watgpu108_64cores.out
#SBATCH --mem=150G
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=2
#SBATCH --time=720
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=128
#SBATCH --mincpus=128
#SBATCH --nodelist=watgpu108
conda run -n vllm_throughput_cpu python run.py \
    --num-warmup-runs 1 \
    --num-runs 3 \
    --models JackFram/llama-68m JackFram/llama-160m huggyllama/llama-7b huggyllama/llama-13b huggyllama/llama-65b \
    --num-concurrent-requests 1 2 4 8 16 32 64 128 \
    --num-input-tokens 1 2 4 8 16 32 64 128 256 \
    --num-output-tokens 1 2 4 8 16 32 64 128 256 \
    --cpu-omp-threads-binds 0-15
