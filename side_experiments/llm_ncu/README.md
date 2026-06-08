# LLM NCU Experiment

In this experiment, we analyze the bottleneck in LLM inference. 

## Notes
* You may need to remove kptr restrictions on your system
    ```
    echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
    echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
    ```

