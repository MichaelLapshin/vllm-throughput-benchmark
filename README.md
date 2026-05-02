# vLLM Throughput Benchmark

This project is for benchmarking the throughput of vLLM on variouns systems (including CPU builds).

## Benchmark Setup
1. Clone the repo
```
git clone --recurse-submodules git@github.com:MichaelLapshin/vllm-throughput-benchmark.git
cd vllm-throughput-benchmark
```
2. Setup conda environment
    - CPU build
```
conda env create -f environment_cpu.yaml
conda activate vllm_throughput_cpu
```
    - GPU build
```
conda env create -f environment_gpu.yaml
conda activate vllm_throughput_gpu
```
3. Install vLLM (environment variables are already set by conda)
   - Note: Compiling would take a while (for the CPU build). Consider increasing the `MAX_JOBS` environment variable, as long as there is enough memory to support more concurrent jobs.
```
cd vllm
python setup.py install
cd ..
pip install -e ./vllm --no-build-isolation
pip install -e ./vllm --no-cache-dir --force-reinstall
```

## Running the Benchmark
1. Review the benchmark parameters under `run_parameters.py`
2. Run the program with `python run.py`
3. Plot the results with `python plot.py --name <results_dir_name>`