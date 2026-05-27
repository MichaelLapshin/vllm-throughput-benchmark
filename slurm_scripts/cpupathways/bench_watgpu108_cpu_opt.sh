#!/bin/bash
#SBATCH --output=slurm_watgpu108_cpu_opt.out
#SBATCH --mem=164G
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
    --num-runs 5 \
    --models facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/opt-30b facebook/opt-66b \
    --num-concurrent-requests 1 2 64 128 \
    --num-input-tokens 1 2 256 \
    --num-output-tokens 1 2 256 \
    --cpu-omp-threads-binds None