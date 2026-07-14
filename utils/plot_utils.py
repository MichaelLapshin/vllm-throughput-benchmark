from typing import List, Any, Callable, Dict, Iterable, Tuple
import csv
from dacite import from_dict
from ast import literal_eval
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import pandas as pd

from results import RequestData, EmissionsData
from run_constants import RESULTS_DIR, PLOTS_DIR
from utils import metadata_util

MARKERS = ['*', '^', 'P', 's', 'v', 'p', 'D', 'X']

def load_csv_emissions(results_dir: str) -> Dict[str, EmissionsData]:
    file_path = f"{results_dir}/emissions.csv"
    if not os.path.isfile(file_path):
        return {}

    emissions: Dict[EmissionsData] = {}
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if key not in EmissionsData.__dataclass_fields__:
                    continue
                if EmissionsData.__dataclass_fields__[key].type == bool:
                    assert row[key] in ["True", "False"]
                    row[key] = row[key] == "True"
                elif EmissionsData.__dataclass_fields__[key].type == int:
                    row[key] = 0 if row[key] == '' else int(row[key])
                elif EmissionsData.__dataclass_fields__[key].type == float:
                    row[key] = 0 if row[key] == '' else float(row[key])
                elif EmissionsData.__dataclass_fields__[key].type == List[float]:
                    row[key] = literal_eval(row[key])
            emissions_obj = from_dict(data_class=EmissionsData, data=row)
            assert emissions_obj.experiment_id not in emissions
            emissions[emissions_obj.experiment_id] = emissions_obj
    return emissions

def load_csv_data(results_dir: str) -> List[RequestData]:
    results: List[RequestData] = []
    with open(f"{results_dir}/data.csv", newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if key not in RequestData.__dataclass_fields__:
                    continue
                if RequestData.__dataclass_fields__[key].type == bool:
                    assert row[key] in ["True", "False"]
                    row[key] = row[key] == "True"
                elif RequestData.__dataclass_fields__[key].type == int:
                    row[key] = int(row[key])
                elif RequestData.__dataclass_fields__[key].type == float:
                    row[key] = float(row[key])
                elif RequestData.__dataclass_fields__[key].type == List[float]:
                    row[key] = literal_eval(row[key])
            results.append(from_dict(data_class=RequestData, data=row))
    return results

def get_model(results: List[RequestData]) -> str:
    model = results[0].model
    assert all(model == r.model for r in results)
    return model

def get_common_metadata(results: List[RequestData], metadata: dict):
    assert len(results) > 0
    assert "parameters" in metadata
    assert "constants" in metadata
    assert "environment" in metadata
    assert isinstance(metadata["environment"]["RUN_ON_CPU"], str)
    assert metadata["environment"]["RUN_ON_CPU"] in ["True", "False"]
    cpu_name = metadata["environment"]["CPU_NAME"]
    run_on_cpu = metadata["environment"]["RUN_ON_CPU"] == "True"
    gpu_name = "<none>" if run_on_cpu else metadata["environment"]["GPU_NAME"]
    models = literal_eval(metadata["parameters"]["PARAM_MODELS"])
    return cpu_name, gpu_name, run_on_cpu, models 

def group_and_find_best_records(
    data: List[RequestData],
    group_by_fn: Callable[[RequestData], Any], # to use for x-axis in plot
    sub_group_by_fn: Callable[[RequestData], Any], # to use a lines in plot
    metric_fn: Callable[[RequestData], Any],
    best_attr_fn: Callable[[RequestData], Any],
    minimize: bool,
) -> tuple[dict, list]:
    """
    Groups data, finds the best record per sub-group based on a metric,
    and filters the original groups to only keep records matching the best configuration.
    """

    # Initial grouping
    groups = defaultdict(list)
    for item in data:
        groups[group_by_fn(item)].append(item)

    # Identify the best record for each sub-group
    best_results: Dict[Any, Dict[Any, Any]] = {}
    for group_key, group in groups.items():
        if group_key not in best_results:
            best_results[group_key] = {}

        for item in group:
            sub_key = sub_group_by_fn(item)
            current_metric = metric_fn(item)
            
            # Check if this is the first time seeing the sub_key, or if it's better (smaller metric)
            if (sub_key not in best_results[group_key] or 
                    (minimize and current_metric < metric_fn(best_results[group_key][sub_key])) or
                    (not minimize and current_metric > metric_fn(best_results[group_key][sub_key]))
                ):
                best_results[group_key][sub_key] = item
    
    # Filter groups and collect the best attributes
    best_attributes = set()
    for group_key, sub_groups in best_results.items():
        for sub_key, best_item in sub_groups.items():
            best_attr_value = best_attr_fn(best_item)
            best_attributes.add(best_attr_value)
            
            # Filter the group to keep items matching the best attribute OR belonging to other sub-keys
            groups[group_key] = list(filter(
                lambda r: best_attr_fn(r) == best_attr_value or sub_group_by_fn(r) != sub_key,
                groups[group_key]
            ))
    
    return dict(groups), sorted(list(best_attributes))

def keep_per_request_batch(
    items: Iterable[RequestData],
    metric_fn: Callable[[RequestData], float],
    keep_max: bool,
) -> List[RequestData]:
    best: Dict[Any, RequestData] = {}
    for item in items:
        key = item.request_batch_uid
        if (key not in best or
            (keep_max and metric_fn(item) > metric_fn(best[key])) or 
            (not keep_max and metric_fn(item) < metric_fn(best[key]))
        ):
            best[key] = item
    return list(best.values())

def get_colour_cycle(n=24):
    cmap = plt.get_cmap('hsv') 
    colors = cmap(np.linspace(0, 1, n))
    return itertools.cycle(colors)
    # return itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def plot_fitted_line(color, x, y) -> str:
    """
    Plot fitted line, return formula
    """
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [m * xv + b for xv in x], color=color, linestyle=':')
    return f"{m:.4f}n + {b:.4f}"

def get_poly_colour_no_alpha(poly):
    r, g, b, _ = poly.get_facecolor()[0]
    color_no_alpha = (r, g, b)
    return color_no_alpha

def format_multisample_data(x: list, y: list) -> Tuple:
    df = pd.DataFrame({'x': x, 'y': y})
    stats = df.groupby('x')['y'].agg(['mean', 'std']).reset_index()
    return (
        np.array(stats['x'].tolist()),
        np.array(stats['mean'].tolist()),
        np.array(stats['std'].tolist())
    )

def int_in_range(num: int, range_str: str) -> bool:
    ranges = range_str.split(',')
    for r in ranges:
        start, end = map(int, r.split('-'))
        if start <= num <= end:
            return True
    return False

def sort_xyz(x, y, z, base_order):
    return map(np.array, zip(*sorted(zip(x, y, z), key=lambda p: base_order.index(p[0]))))
