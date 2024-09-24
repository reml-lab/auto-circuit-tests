from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
import random
from copy import deepcopy
from functools import partial

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

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, compute_scores, DIV_ANSWER_FUNCS
from auto_circuit_tests.edge_graph import SeqGraph, sample_paths 
from auto_circuit_tests.hypo_tests.utils import join_values, remove_el
from auto_circuit_tests.utils.auto_circuit_utils import run_circuit_with_edge_ablated


class MinResult(NamedTuple):
    num_edge_score_gt_ref: int
    n: int
    reject_min_null: bool
    reject_null_non_min: bool 
    p_value_min: float
    p_value_non_min: float


def min_test(
    k: int,
    n: int,
    q_star: float,
    alpha: float,
    null_minimal: bool,
) -> Tuple[bool, float]:
    p_value = binom.cdf(k, n, q_star) if null_minimal else 1 - binom.cdf(k, n, q_star)
    return bool(p_value < alpha), p_value


def score_diffs(
    dataloader: PromptDataLoader,
    outs_1: BatchOutputs, 
    outs_2: BatchOutputs,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    model_outs: Optional[BatchOutputs] = None,
    device: str = t.device('cuda')
) -> list[t.Tensor]:
    diffs = []
    score_func = partial(compute_scores, grad_func=grad_func, answer_func=answer_func)
    for batch in dataloader:
        batch: PromptPairBatch
        model_outs_batch = model_outs[batch.key].to(device) if model_outs is not None else None
        score_1 = score_func(outs_1[batch.key].to(device), batch, model_outs_batch)
        score_2 = score_func(outs_2[batch.key].to(device), batch, model_outs_batch)
        diffs.append(t.abs(score_1 - score_2).detach().cpu())
    return diffs

def minimality_test_edge(
    ablated_edge_mean_diff: float,
    inflated_ablated_mean_diffs: list[float],
    n_edges: int,
    alpha: float = 0.05, 
    q_star: float = 0.9,
) -> MinResult: 
    n = len(inflated_ablated_mean_diffs)
    k = sum(
        ablated_edge_mean_diff > inflated_ablated_mean_diff
        for inflated_ablated_mean_diff in inflated_ablated_mean_diffs
    )
    # min null (bonferroni correction)
    reject_min_null, p_value_min = min_test(
        k, n, q_star, alpha / n_edges, null_minimal=True
    )
    # non-min null (no bonferroni correction)
    reject_non_min_null, p_value_non_min = min_test(
        k, n, q_star, alpha , null_minimal=False
    )
    return MinResult(
        num_edge_score_gt_ref=k,
        n=n,
        reject_min_null=reject_min_null,
        reject_null_non_min=reject_non_min_null,
        p_value_min=p_value_min,
        p_value_non_min=p_value_non_min
    )


def rem_edge_from_paths(paths: list[list[Edge]]) -> list[list[Edge]]:
    return [remove_el(path, random.choice(range(len(path)))) for path in paths]


def run_circuits_inflated_ablated(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    ablation_type: AblationType,
    edges: list[Edge],
    n_paths: Optional[int] = None,
    graph: Optional[SeqGraph] = None,
    paths: Optional[list[list[Edge]]] = None,
    ablated_paths: Optional[list[list[Edge]]] = None,
    token: Optional[bool] = None
) -> Tuple[CircuitOutputs, CircuitOutputs]:
    # build graph
    if graph is None:
        assert token is not None
        graph = SeqGraph(model.edges, token=token, attn_only=model.cfg.attn_only)
    
    # sample paths
    if paths is None:
        complement_edges = set(model.edges) - set(edges)
        paths = sample_paths(
            seq_graph=graph, 
            n_paths=n_paths,
            complement_edges=complement_edges,
        )
        edges_set = set(edges)
        novel_edge_paths = [[edge for edge in path if edge not in edges_set] for path in paths]

    # run inflated circuits
    inflated_outs: CircuitOutputs = {}
    for i, path in tqdm(enumerate(paths), desc="Inflated Circuits", total=len(paths)): 
        circ_edges = set(path + edges)
        prune_scores = model.circuit_prune_scores(circ_edges)
        inflated_out = next(iter(run_circuits(
            model=model, 
            dataloader=dataloader,
            prune_scores=prune_scores,
            test_edge_counts=[len(circ_edges)],
            patch_type=PatchType.TREE_PATCH, 
            ablation_type=ablation_type,
            reverse_clean_corrupt=False, 
        ).values()))
        inflated_outs[i] = {k: v.detach().cpu() for k, v in inflated_out.items()}
    
    # run ablated circuits
    if ablated_paths is None:
        ablated_paths = rem_edge_from_paths(novel_edge_paths)
    ablated_outs: CircuitOutputs = {}
    for i, path in tqdm(enumerate(ablated_paths), desc="Ablated Circuits", total=len(ablated_paths)):
        circ_edges = set(path + edges)
        prune_scores = model.circuit_prune_scores(circ_edges)
        ablated_out = next(iter(run_circuits(
            model=model, 
            dataloader=dataloader,
            prune_scores=prune_scores,
            test_edge_counts=[len(circ_edges)],
            patch_type=PatchType.TREE_PATCH, 
            ablation_type=ablation_type,
            reverse_clean_corrupt=False, 
        ).values()))
        ablated_outs[i] = {k: v.detach().cpu() for k, v in ablated_out.items()}

    return inflated_outs, ablated_outs
    

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
    circuit_outs: Optional[BatchOutputs]=None,
    edges_outs: Optional[Dict[Edge, BatchOutputs]]=None, 
    model_outs: Optional[BatchOutputs]=None,
    inflated_outs: Optional[CircuitOutputs]=None,
    ablated_outs: Optional[CircuitOutputs]=None,
    n_paths: Optional[int] = None,
    alpha: float = 0.05, 
    q_star: float = 0.9,
    device: str = t.device('cuda'),
    stop_if_reject: bool = False
) -> Tuple[Dict[Edge, MinResult], Optional[bool]]:
    
    assert (inflated_outs is None) == (ablated_outs is None) == (n_paths is not None)
    
    # model outs (for div metrics)
    if answer_func in DIV_ANSWER_FUNCS and model_outs is None:
        model_outs = {
            batch.key: model(batch.clean)[model.out_slice]
            for batch in dataloader
        }

    # inflated ablated 
    if inflated_outs is None:
        inflated_outs, ablated_outs = run_circuits_inflated_ablated(
            model=model,
            dataloader=dataloader,
            ablation_type=ablation_type,
            edges=edges,
            n_paths=n_paths,
            token=token
        )

    # compute mean diffs for each inflated circuit / ablated circuit
    inflated_ablated_mean_diffs: list[float] = []
    for i, inflated_out in inflated_outs.items():
        inflated_ablated_diffs = score_diffs(
            dataloader=dataloader,
            outs_1=inflated_out,
            outs_2=ablated_outs[i],
            grad_func=grad_func,
            answer_func=answer_func,
            device=device
        )
        inflated_ablated_mean_diffs.append(t.cat(inflated_ablated_diffs).mean().item())

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

    
    # edge score differences
    def get_edge_out(edge: Edge) -> BatchOutputs:
        return run_circuit_with_edge_ablated(
            model=model,
            dataloader=dataloader,
            prune_scores=prune_scores,
            edge=edge,
            ablation_type=ablation_type,
            threshold=threshold,
            to_cpu=True
        )
    def get_ablated_edge_mean_diff(edge_out: BatchOutputs) -> float:
        ablated_diffs = score_diffs(
            dataloader=dataloader,
            outs_1=edge_out,
            outs_2=circuit_outs,
            grad_func=grad_func,
            answer_func=answer_func,
            device=device
        )
        return t.cat(ablated_diffs).mean().item()  
    
    ablated_edge_mean_diffs: dict[Edge, float] = {}
    # edges out 
    if edges_outs is not None:
        # compute mean diffs for each ablated edge
        for edge in edges:
            ablated_diffs = get_ablated_edge_mean_diff(edges_outs[edge])
            ablated_edge_mean_diffs[edge] = t.cat(ablated_diffs).mean().item()

    # run minimality test
    min_results: dict[Edge, MinResult] = {}
    for edge in tqdm(edges):
        # dynamically compute edge out (saved time if stop_if_reject)
        if edge not in ablated_edge_mean_diffs:
            edge_out = get_edge_out(edge)
            ablated_edge_mean_diffs[edge] = get_ablated_edge_mean_diff(edge_out)
        # run minimality test
        min_results[edge] = minimality_test_edge(
            ablated_edge_mean_diff=ablated_edge_mean_diffs[edge],
            inflated_ablated_mean_diffs=inflated_ablated_mean_diffs,
            n_edges=len(edges),
            alpha=alpha,
            q_star=q_star
        )
        print(min_results[edge].num_edge_score_gt_ref)
        if stop_if_reject and min_results[edge].reject_min_null:
            break
    
    reject_min = any(r.reject_min_null for r in min_results.values())
    return min_results, reject_min


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