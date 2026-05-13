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
      - Note: Compiling would take a while (for the CPU build). Consider increasing the `MAX_JOBS` environment variable, as long as there is enough memory to support more concurrent jobs.
    ```
    conda env create -f environment_cpu.yaml
    conda activate vllm_throughput_cpu
    conda env config vars set LD_PRELOAD="$CONDA_PREFIX/lib/libtcmalloc.so:$CONDA_PREFIX/lib/libiomp5.so:$LD_PRELOAD" -n vllm_throughput_cpu
    conda deactivate vllm_throughput_cpu
    conda activate vllm_throughput_cpu
    cd vllm_cpu
    python setup.py install
    cd ..
    pip install -e ./vllm_cpu --no-build-isolation
    ```
    - GPU build
    ```
    conda env create -f environment_gpu.yaml
    conda activate vllm_throughput_gpu
    conda env config vars set LD_PRELOAD="$CONDA_PREFIX/lib/libtcmalloc.so:$CONDA_PREFIX/lib/libiomp5.so:$LD_PRELOAD" -n vllm_throughput_gpu
    conda deactivate vllm_throughput_gpu
    conda activate vllm_throughput_gpu
    cd vllm_gpu
    python setup.py install
    cd ..
    pip install -e ./vllm_gpu --no-build-isolation
    ```
3. (Optional) Set HuggingFace API token to benchmark gated models
    ```
    echo "HF_TOKEN = \"<api_token>\"" > .env
    ```

## Running the Benchmark
1. Review the benchmark parameters under `run_parameters.py`
2. Run the program with `python run.py`
3. Plot the results with `python plot.py --name <results_dir_name>`

## Helpful Commands
Restore git submodules:
```
git submodule update --init --recursive
```
Delete conda CPU environment
```
conda env remove -n vllm_throughput_cpu
```