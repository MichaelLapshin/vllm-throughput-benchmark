"""
This file defines constants

Some of these variables are based on assumptions,
so there may be cases where one would like to change them.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directories
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Value of zero indicates greedy sampling
VLLM_SAMPLING_TEMPERATURE = 0
