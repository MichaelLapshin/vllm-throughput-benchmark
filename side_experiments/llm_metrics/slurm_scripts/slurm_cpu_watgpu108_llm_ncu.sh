#!/bin/bash
#SBATCH --output=slurm_cpu_watgpu108_llm_metrics.out
#SBATCH --mem=150G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=2
#SBATCH --time=720
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=64
#SBATCH --mincpus=64
#SBATCH --nodelist=watgpu108
newgrp perf_users && \
conda run -n vllm_throughput_cpu python -m side_experiments.llm_metrics.run_profiler \
    --models JackFram/llama-68m JackFram/llama-160m Qwen/Qwen3-0.6B Qwen/Qwen3-4B huggyllama/llama-7b huggyllama/llama-13b mistralai/Codestral-22B-v0.1 \
    --num-output-tokens 1024 \
    --schedulers NoSpecDecScheduler_Sequential \
    --perf-stat-runs 2 \
    --perf-stat-profile-metrics
