# NOTE: Modify `common_config.py` to set the parameters for the following scripts.

python -m side_experiments.llm_metrics.run_profiler --perf-stat-profile-metrics
python -m side_experiments.llm_metrics.run_profiler --profile-gpu
python -m side_experiments.llm_metrics.plot_ncu_data
