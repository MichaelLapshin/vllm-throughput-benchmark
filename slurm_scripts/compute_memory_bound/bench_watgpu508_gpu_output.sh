#!/bin/bash
#SBATCH --output=slurm_bench_watgpu508_gpu.out
#SBATCH --mem=150G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=2
#SBATCH --time=960
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --mincpus=16
#SBATCH --nodelist=watgpu508
conda run -n vllm_throughput_gpu \
    python run.py \
    --num-warmup-runs 1 \
    --num-runs 3 \
    --models Qwen/Qwen3-8B \
    --num-concurrent-requests 1 \
    --num-input-tokens 1 \
    --num-output-tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32767 \
    --cpu-omp-threads-binds None
