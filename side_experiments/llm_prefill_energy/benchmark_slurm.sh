#!/bin/bash
#SBATCH --output=side_experiments/prefill_energy/slurm_task.out
#SBATCH --mem=32G
#SBATCH --sockets-per-node=1
#SBATCH --threads-per-core=2
#SBATCH --time=720
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --mincpus=16
conda run -n vllm_throughput_gpu python -m side_experiments.prefill_energy.run
