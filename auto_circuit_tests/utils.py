import os
import json
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime
from typing import Dict, Any

import torch

from auto_circuit.types import AblationType
from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc


def repo_path_to_abs_path(path: str) -> Path:
    """
    (from auto-circuit)
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.absolute()
    return repo_abs_path / path

OUTPUT_DIR = Path("output")
RESULTS_DIR = repo_path_to_abs_path(OUTPUT_DIR / "hypo_test_results")

def save_cache(data_dict: Dict[Any, Any], folder_name: str, base_filename: str):
    """
    (from auto-circuit)
    Save a dictionary to a cache file.

    Args:
        data_dict: The dictionary to save.
        folder_name: The name of the folder to save the cache in.
        base_filename: The base name of the file to save the cache in. The current date
            and time will be appended to the base filename.
    """
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = folder / f"{base_filename}-{dt_string}.pkl"
    print(f"Saving cache to {file_path}")
    torch.save(data_dict, file_path)


def load_cache(folder_name: str, filename: str) -> Dict[Any, Any]:
    """
    (from auto-circuit)
    Load a dictionary from a cache file.

    Args:
        folder_name: The name of the folder to load the cache from.
        filename: The name of the file to load the cache from.

    Returns:
        The loaded dictionary.
    """
    folder = repo_path_to_abs_path(folder_name)
    return torch.load(folder / filename)

def save_json(data, folder_name: str, base_filename: str): 
    """
    Save data to a json file.

    Args:
        data: The data to save.
        filename: The name of the file to save the data in.
    """
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{base_filename}.json"
    print(f"Saving json to {file_path}")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(folder_name: str, filename: str):
    """
    Load data from a json file.

    Args:
        filename: The name of the file to load the data from.

    Returns:
        The loaded data.
    """
    folder = repo_path_to_abs_path(folder_name)
    with open(folder / filename, "r") as f:
        return json.load(f)


def get_exp_dir(
    task_key: str,
    ablation_type: AblationType,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    ig_samples: int, 
    layerwise: bool,
    act_patch: bool,
    use_abs: bool,
    alpha: float,
    epsilon: float,
    q_star: float
):
    # handle directories
    out_dir = RESULTS_DIR
    # should resctructure output directory to be task/ablation_type/output_func_answer_func/prune_score/experiments
    task_dir = out_dir / task_key.replace(' ', '_')
    ablation_dir = task_dir / ablation_type.name
    out_answer_dir = ablation_dir / f"{grad_func.name}_{answer_func.name}"
    ps_dir = out_answer_dir / (f"{ig_samples}_{layerwise}" if not act_patch else "act_patch")
    exp_dir = ps_dir / f"{use_abs}_{alpha}_{epsilon}_{q_star}"
    return task_dir, ablation_dir, out_answer_dir, ps_dir, exp_dir