# LLM NCU Experiment

In this experiment, we analyze the bottleneck in LLM inference. 

## Notes
* You may need to remove kptr restrictions on your system
    ```
    echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
    echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
    ```

## Obtaining all Profiling Events
```
ncu --list-metrics > ncu_list_metrics.txt
ncu --query-metrics-mode all > ncu_list_metrics_all.txt
perf list --desc --long-desc --details > perf_list.txt
```

## Launching vLLM NCU
```
python -m side_experiments.llm_metrics.run_profiler \
--profiler-type ncu_profiler \
--models JackFram/llama-68m JackFram/llama-160m huggyllama/llama-7b huggyllama/llama-13b mistralai/Codestral-22B-v0.1 \
--num-output-tokens 1 \
--schedulers NoSpecDecScheduler_Sequential
```

```
python -m side_experiments.llm_metrics.plot_ncu_data -n <results_dir>
```

## Viewing Results
```
perf mem report --sort=mem -i <report_file>
```

## watgpu108 salloc command
```
salloc --nodelist=watgpu108 --mem=100G --sockets-per-node=1 --cores-per-socket=32 --threads-per-core=2 --gpus=0 --ntasks=1 --ntasks-per-core=1 --ntasks-per-socket=1 --cpus-per-task=64 --mincpus=64
```