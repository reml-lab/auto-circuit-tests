from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
import random
from copy import deepcopy

import torch as t
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

from auto_circuit.data import PromptDataLoader, PromptPairBatch
from auto_circuit.prune import run_circuits
from auto_circuit.utils.tensor_ops import prune_scores_threshold
from auto_circuit.types import (
    CircuitOutputs, 
    BatchKey,
    PruneScores,
    BatchOutputs,
    PatchType, 
    AblationType,
    Edge
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.custom_tqdm import tqdm

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, get_score_func
from auto_circuit_tests.edge_graph import SeqGraph, sample_paths 
from auto_circuit_tests.hypo_tests.utils import join_values, remove_el
from auto_circuit_tests.utils.auto_circuit_utils import run_circuit_with_edges_ablated




class MinResult(NamedTuple):
    num_edge_score_gt_ref: int
    n: int
    null_minimal: bool
    reject_null: bool
    p_value: float
    diffs: t.Tensor


def min_test(
    k: int,
    n: int,
    q_star: float,
    alpha: float,
    null_minimal: bool,
) -> Tuple[bool, float]:
    p_value = binom.cdf(k, n, q_star) if null_minimal else 1 - binom.cdf(k, n, q_star)
    return p_value < alpha, p_value


def score_diffs(
    dataloader: PromptDataLoader,
    outs_1: BatchOutputs, 
    outs_2: BatchOutputs,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    device: str = t.device('cuda')
) -> list[t.Tensor]:
    diffs = []
    score_func = get_score_func(grad_func, answer_func)
    for batch in dataloader:
        batch: PromptPairBatch
        score_1 = score_func(outs_1[batch.key].to(device), batch)
        score_2 = score_func(outs_2[batch.key].to(device), batch)
        diffs.append(t.abs(score_1 - score_2).detach().cpu())
    return diffs
    

def minimality_test_edge(
    model: PatchableModel, 
    dataloader: PromptDataLoader,
    edge_outs: BatchOutputs, 
    circuit_outs: BatchOutputs,
    inflated_ablated_mean_diff: float,
    # circuit_outs_inflated: BatchOutputs, 
    # circuit_outs_ablated: BatchOutputs,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    null_minimal: bool = True,
    alpha: float = 0.05, 
    q_star: float = 0.9,
    device: str = t.device('cuda')
) -> MinResult: 
    score_func = get_score_func(grad_func, answer_func)
    n = 0 
    k = 0
    diffs = []
    for batch in dataloader:
        batch: PromptPairBatch

        edge_score = score_func(edge_outs[batch.key].to(device), batch)
        circ_score = score_func(circuit_outs[batch.key].to(device), batch)
        edge_diff = t.abs(edge_score - circ_score)

        # ablated_score = score_func(circuit_outs_ablated[batch.key].to(device), batch)
        # inflated_score = score_func(circuit_outs_inflated[batch.key].to(device), batch)
        # edge_diff_inflated = t.abs(ablated_score - inflated_score)

        diffs.append(edge_diff.detach().cpu())
        # diffs_inflated.append(edge_diff_inflated.detach().cpu())
        k += t.sum(edge_diff > inflated_ablated_mean_diff).item()
        n += batch.clean.size(0)
    
    reject_null, p_value = min_test(k, n, q_star, alpha, null_minimal)
    return MinResult(k, n, null_minimal, reject_null, p_value, t.cat(diffs))


# run circuits with paths added 
def _new_instance_prune_scores(
    model: PatchableModel, 
    batch_size: int, 
    init_val: float=0.0,
    prune_scores: Optional[PruneScores]=None,
) -> PruneScores: 
    instance_prune_scores: PruneScores = {}
    if prune_scores is not None: # repeat prune scores for each instance in batch
        for mod_name, mask in prune_scores.items():
            instance_prune_scores[mod_name] = prune_scores[mod_name].unsqueeze(0).repeat_interleave(batch_size, dim=0)
        return instance_prune_scores
    for (mod_name, mask) in model.patch_masks.items():
        instance_prune_scores[mod_name] = t.full((batch_size, *mask.shape), init_val)
    return prune_scores

def make_inflated_batch_prune_scores(
    model: PatchableModel,
    prune_scores: PruneScores, 
    dataloader: PromptDataLoader,
    paths: list[list[Edge]],
    batch_size: int,
    threshold: float
) -> Dict[BatchKey, PruneScores]:
    inflated_batch_prune_scores: Dict[BatchKey, PruneScores] = {}
    for batch_count, batch in enumerate(dataloader): 
        batch_prune_scores = _new_instance_prune_scores(model, batch_size, prune_scores=prune_scores)
        for i in range(batch_size):
            path = paths[batch_count * batch_size + i]
            # add path to prune scores
            for edge in path:
                batch_prune_scores[edge.dest.module_name][i][edge.patch_idx] = threshold
        inflated_batch_prune_scores[batch.key] = batch_prune_scores
    return inflated_batch_prune_scores


def _rem_edge_from_paths(paths: list[list[Edge]]) -> list[list[Edge]]:
    return [remove_el(path, random.choice(range(len(path)))) for path in paths]


def run_circuits_inflated_ablated(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    threshold: float,
    ablation_type: AblationType,
    edges: list[Edge],
    graph: Optional[SeqGraph] = None,
    paths: Optional[list[list[Edge]]] = None,
    ablated_paths: Optional[list[list[Edge]]] = None,
    token: Optional[bool] = None
) -> Tuple[BatchOutputs, BatchOutputs]:
    # build graph
    if graph is None:
        assert token is not None
        graph = SeqGraph(model.edges, token=token, attn_only=model.cfg.attn_only)
    
    # TODO: the test is comparing expected values over the dataset, so instead of sampling 
    # a new path and ablation per instance, we should sample n=200 paths and ablations, and compare their 
    # perforamnce across the entire dataset
    
    
    # sample paths
    if paths is None:
        complement_edges = set(model.edges) - set(edges)
        n_paths = dataloader.batch_size * len(dataloader)
        paths = sample_paths(
            seq_graph=graph, 
            n_paths=n_paths,
            complement_edges=complement_edges,
        )
        edges_set = set(edges)
        novel_edge_paths = [[edge for edge in path if edge not in edges_set] for path in paths]

    # run inflated circuits
    inflated_batch_prune_scores = make_inflated_batch_prune_scores(
        model=model,
        prune_scores=prune_scores,
        dataloader=dataloader,
        paths=paths,
        batch_size=dataloader.batch_size,
        threshold=threshold,
    )

    circuit_out_inflated: BatchOutputs = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        prune_scores=inflated_batch_prune_scores,
        thresholds = [threshold],
        patch_type=PatchType.TREE_PATCH, 
        ablation_type=ablation_type,
        reverse_clean_corrupt=False, 
    ))

    # run ablated circuits
    if ablated_paths is None:
        ablated_paths = _rem_edge_from_paths(novel_edge_paths)
    ablated_batch_prune_scores = make_inflated_batch_prune_scores(
        model=model,
        prune_scores=prune_scores,
        dataloader=dataloader,
        paths=ablated_paths,
        batch_size=dataloader.batch_size,
        threshold=threshold,
    )

    circuit_out_ablated: BatchOutputs = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        prune_scores=ablated_batch_prune_scores,
        thresholds = [threshold],
        patch_type=PatchType.TREE_PATCH, 
        ablation_type=ablation_type,
        reverse_clean_corrupt=False, 
    ))
    return circuit_out_inflated, circuit_out_ablated
    

def minimality_test(
    model: PatchableModel, 
    dataloader: PromptDataLoader, 
    edges: list[Edge],
    prune_scores: PruneScores,
    threshold: float,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    ablation_type: AblationType,
    token: bool,
    circuit_outs: Optional[CircuitOutputs]=None,
    edges_outs: Optional[Dict[Edge, BatchOutputs]]=None, 
    circuit_out_inflated: Optional[CircuitOutputs]=None,
    circuit_out_ablated: Optional[CircuitOutputs]=None,
    null_minimal: bool = True,
    alpha: float = 0.05, 
    bonferonni: bool = False,
    q_star: float = 0.9,
    device: str = t.device('cuda')
) -> Tuple[Dict[Edge, MinResult], bool]:
    
    assert (circuit_out_inflated is None) == (circuit_out_ablated is None)
    if bonferonni:
        alpha = alpha / len(edges)

    # TODO: the test is comparing expected values over the dataset, so instead of sampling 
    # a new path and ablation per instance, we should sample n=200 paths and ablations, and compare their 
    # perforamnce across the entire dataset
    
    # circuit outs
    if circuit_outs is None:
        circuit_outs = next(iter(run_circuits(
            model=model, 
            dataloader=dataloader,
            prune_scores=prune_scores,
            test_edge_counts=[len(edges)],
            patch_type=PatchType.TREE_PATCH, 
            ablation_type=ablation_type,
            reverse_clean_corrupt=False, 
        ).values()))
    
    # edges out 
    if edges_outs is None:
        edges_outs = run_circuit_with_edges_ablated(
            model=model,
            dataloader=dataloader,
            edges=edges,
            prune_scores=prune_scores,
            ablation_type=ablation_type,
            threshold=threshold,
            to_cpu=True
        )
    # inflated ablated 
    if circuit_out_inflated is None:
        circuit_out_inflated, circuit_out_ablated = run_circuits_inflated_ablated(
            model=model,
            dataloader=dataloader,
            prune_scores=prune_scores,
            threshold=threshold,
            ablation_type=ablation_type,
            edges=edges,
            graph=None,
            token=token
        )
    # compute mean diff
    inflated_ablated_diffs = score_diffs(
        dataloader=dataloader,
        outs_1=circuit_out_inflated,
        outs_2=circuit_out_ablated,
        grad_func=grad_func,
        answer_func=answer_func,
        device=device
    )
    inflated_ablated_mean_diff = t.cat(inflated_ablated_diffs).mean().item()

    # run minimality test
    min_results = {}
    for edge in tqdm(edges):
        min_results[edge] = minimality_test_edge(
            model=model,
            dataloader=dataloader,
            edges=[edge],
            edge_outs=edges_outs[edge],
            circuit_outs=circuit_outs,
            inflated_ablated_mean_diff=inflated_ablated_mean_diff,
            grad_func=grad_func,
            answer_func=answer_func,
            null_minimal=null_minimal,
            alpha=alpha,
            q_star=q_star,
            device=device
        )
    return min_results, all(r.reject_null for r in min_results.values())


def plot_p_values(min_results: dict[Edge, MinResult], edge_scores: dict[Edge, t.Tensor], alpha: float = 0.05):
    fig, ax = plt.subplots(figsize=(12, 2))
    p_values = [r.p_value for r in min_results.values()]
    neg_edge = [edge_scores[edge].cpu() < 0 for edge in min_results.keys()]
    ax.scatter(range(len(p_values), 0, -1), p_values, c=neg_edge, cmap='coolwarm')
    ax.set_xlim(len(p_values), 0)
    # plot alpha line 
    ax.axhline(y=alpha, color='g', linestyle='-')
    ax.set_title("p values for minimality test")
    return fig, ax

def plot_edge_k(min_results: dict[Edge, MinResult], edge_scores: dict[Edge, t.Tensor], n: int, q_star: float):
    fig, ax = plt.subplots(figsize=(12, 2))
    ks = [r.num_edge_score_gt_ref for r in min_results.values()]
    neg_edge = [edge_scores[edge].cpu() < 0 for edge in min_results.keys()]
    # scatter with blue as positive, red as negative
    ax.scatter(range(len(ks), 0, -1), ks, c=neg_edge, cmap='coolwarm')
    ax.set_xlim(len(ks), 0)
    # horizontal line at  
    ax.axhline(y=n // 2, color='g', linestyle='--', label=f"N / 2")
    # horizeontal line at n * q_star
    ax.axhline(y=n * q_star, color='r', linestyle='--', label=f"N x q* ({q_star})")

    ax.set_title("k for minimality test")

    ax.legend()
    return fig, ax

def plot_score_quantiles(
    min_results: dict[Edge, MinResult],
    edge_scores: dict[Edge, t.Tensor],
    quantile_range: list[float] = [0.00, 1.00]
):
    # calculate quantiles 
    quantiles = [
        np.quantile(min_results[edge].diffs.numpy(), quantile_range) 
        for edge in min_results.keys()
    ]
    lower_quantiles = [q[0] for q in quantiles]
    upper_quantiles = [q[1] for q in quantiles]

    # compute mean and quartiles of diff inflated
    diff_infl = t.cat([r.diffs_inflated for r in min_results.values()])
    quantile_infl = np.quantile(diff_infl.numpy(), quantile_range)
    mean_infl = diff_infl.mean().numpy()
    median_infl = diff_infl.median().numpy()

    # plot average diff with quantile ranges
    fig, ax = plt.subplots(figsize=(12, 4))
    diffs = [r.diffs.mean().numpy() for r in min_results.values()]
    median_diffs = [r.diffs.median().numpy() for r in min_results.values()]

    # Plot error bars with quantile ranges, median, and mean
    ax.errorbar(range(len(diffs), 0, -1), diffs, 
                yerr=[np.array(diffs) - lower_quantiles, upper_quantiles - np.array(diffs)],
                fmt='none', capsize=5, capthick=1)

    # Add median points in orange
    ax.scatter(range(len(median_diffs), 0, -1), median_diffs, color='orange', marker='s', s=30, label='Median', zorder=3)

    # Add mean points in green
    ax.scatter(range(len(diffs), 0, -1), diffs, color='green', marker='o', s=30, label='Mean', zorder=3)

    ax.set_xlim(len(diffs), 0)

    # inflated mean and median lines
    ax.axhline(y=mean_infl, color='g', linestyle='-')
    ax.axhline(y=median_infl, color='orange', linestyle='-')

    # Add quantile inflation lines
    ax.axhline(y=quantile_infl[0], color='c', linestyle='--',  zorder=2, label=f'Inflated Quantile Range ({quantile_range[0]*100})')
    ax.axhline(y=quantile_infl[1], color='m', linestyle='--', zorder=2, label=f'Inflated Quantile Range ({quantile_range[1]*100})')

    ax.set_yscale('log')
    ax.set_title(f"Score diff for minimality test (with {quantile_range[0]*100}-{quantile_range[1]*100} quantile ranges)")
    ax.set_xlabel("Edges")
    ax.set_ylabel("Score Difference")

    # Add legend
    ax.legend()
    return fig, ax