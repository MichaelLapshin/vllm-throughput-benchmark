from typing import List, Any, Callable, Dict
import csv
from dacite import from_dict
from ast import literal_eval
from collections import defaultdict

from results import RequestData
from run_constants import RESULTS_DIR, PLOTS_DIR
from utils import metadata_util

MARKERS = ['*', '^', 'P', 's', 'v', 'p']

def load_metadata(results_name: str) -> dict:
    return metadata_util.load_metadata(f"{RESULTS_DIR}/{results_name}")

def load_csv_data(results_name: str) -> List[RequestData]:
    results: List[RequestData] = []
    with open(f"{RESULTS_DIR}/{results_name}/data.csv", newline='', encoding='utf-8') as file:
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
                    current_metric < metric_fn(best_results[group_key][sub_key])):
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