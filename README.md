# vLLM Throughput Benchmark

This project is for benchmarking the throughput of vLLM on variouns systems (including CPU builds).

## Benchmark Setup
1. Clone the repo
```
git clone --recurse-submodules git@github.com:MichaelLapshin/vllm-throughput-benchmark.git
cd vllm-throughput-benchmark
```
1. Setup python environment
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
1. Install caching (if `which ccache` returns nothing)
```
apt install ccache
```
or 
```
conda install ccache
```
2. Install the vLLM instance GPU editable mode
```
pip install -r vllm_gpu/requirements/common.txt
pip install -r vllm_gpu/requirements/cuda.txt
pip install -r vllm_gpu/requirements/build/cuda.txt
VLLM_USE_PRECOMPILED=1 pip install -e vllm_gpu/
```
1. Install the vLLM instance CPU editable mode
```
pip install -r vllm_cpu/requirements/common.txt --torch-backend cpu --index-strategy unsafe-best-match
pip install -r vllm_cpu/requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
pip install -r vllm_cpu/requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
VLLM_TARGET_DEVICE=cpu python vllm_cpu/setup.py install
VLLM_USE_PRECOMPILED=1 VLLM_TARGET_DEVICE=cpu pip install -e vllm_cpu/
```


## Running the Benchmark
1. Review the benchmark parameters under `run_parameters.py`
2. Run the program with `python run.py`
3. Plot the results with `python plot.py`