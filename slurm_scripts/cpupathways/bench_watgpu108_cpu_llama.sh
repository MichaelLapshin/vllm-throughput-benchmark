#!/bin/bash
#SBATCH --output=slurm_watgpu108_cpu_llama2.out
#SBATCH --mem=180G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=2
#SBATCH --time=1440
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=64
#SBATCH --mincpus=64
#SBATCH --nodelist=watgpu108
conda run -n vllm_throughput_cpu python run.py \
    --num-warmup-runs 1 \
    --num-runs 3 \
    --models JackFram/llama-68m JackFram/llama-160m huggyllama/llama-7b huggyllama/llama-13b huggyllama/llama-65b \
    --num-concurrent-requests 1 2 64 128 \
    --num-input-tokens 1 2 256 \
    --num-output-tokens 1 2 256 \
    --cpu-omp-threads-binds None