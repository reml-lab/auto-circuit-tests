from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
import random

import torch 
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

from auto_circuit.data import PromptDataLoader
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
from auto_circuit_tests.hypo_tests.utils import edges_from_mask,  set_score
from auto_circuit_tests.utils.utils import edge_key



class MinResult(NamedTuple):
    not_minimal: bool
    p_value: float
    num_edge_score_gt_ref: int
    diffs: torch.Tensor
    diffs_inflated: torch.Tensor

def join_values(d: Dict[Any, Dict]):
    return {k: v for sub_d in d.values() for k, v in sub_d.items()}


def minimality_test( #TODO: seperate infalted circuit seperate from dataset, get higher n 
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: torch.Tensor | PruneScores,
    edges: list[Edge], 
    edge_count: int, 
    ablation_type: AblationType, 
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    filtered_paths: Optional[list[list[Edge]]] = None,
    seq_graph: Optional[SeqGraph] = None,
    n_paths: Optional[int] = None,
    circuit_out: Optional[CircuitOutputs] = None,
    threshold: Optional[float] = None,
    use_abs: bool = True,
    tokens: bool = False,
    alpha: float = 0.05,
    q_star: float = 0.9,
    max_edges_in_order: int = 0,
    max_edges_in_order_without_fail: int = 0,
) -> Dict[Edge, MinResult]:
    if circuit_out is None:
        circuit_out = dict(next(iter(run_circuits(
            model=model, 
            dataloader=dataloader,
            test_edge_counts=[edge_count],
            prune_scores=prune_scores,
            patch_type=PatchType.TREE_PATCH,
            ablation_type=ablation_type,
            reverse_clean_corrupt=False,
            use_abs=use_abs
        ).values())))
    if threshold is None:
        threshold = prune_scores_threshold(prune_scores, edge_count, use_abs=use_abs)

    # sample filtered paths if not provided
    if filtered_paths is None:
        if seq_graph is None:
            raise ValueError("seq_graph must be provided if filtered_paths is not provided")
        if n_paths is None:
            raise ValueError("n_paths must be provided if filtered_paths is not provided")
        filtered_paths = sample_paths(
            seq_graph=seq_graph,
            n_paths=n_paths,
            tested_edges=edges,
        )
    
    # sample random paths, inflate prune scores, and run
    sampled_paths: Dict[BatchKey, list[Edge]] = {}
    prune_scores_inflated: Dict[BatchKey, PruneScores] = {}
    for batch in dataloader:
        prune_scores_inflated[batch.key] = {
            k: score.unsqueeze(0).repeat_interleave(batch.clean.size(0), 0)
            for k, score in prune_scores.items()
        }
        sampled_paths[batch.key] = random.choices(filtered_paths, k=batch.clean.size(0))
    for batch_key, paths in sampled_paths.items():
        for batch_idx, path in enumerate(paths):
            for edge in path:
                set_score(edge, prune_scores_inflated[batch_key], threshold+1, batch_idx=batch_idx, tokens=tokens)
    
    # join values b/c number of edges can vary by batch
    circuit_out_inflated: BatchOutputs = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_inflated,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
        use_abs=use_abs
    ))

    # ablate random edges in paths and run 
    edges_set = set(edges)
    prune_scores_ablated_paths = prune_scores_inflated
    for batch_key, paths in sampled_paths.items():
        for batch_idx, path in enumerate(paths):
            edge_to_ablate = random.choice(list(set(path) - edges_set)) 
            set_score(edge_to_ablate, prune_scores_ablated_paths[batch_key], 0.0, batch_idx, tokens=tokens)
    circuit_out_ablated_paths: BatchOutputs = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_ablated_paths,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
        use_abs=use_abs
    ))
    # test edges
    def test_edge(edge):
        return minimality_test_edge(
            model=model,
            dataloader=dataloader,
            prune_scores=prune_scores,
            edge=edge,
            circuit_out_inflated=circuit_out_inflated,
            circuit_out_ablated_paths=circuit_out_ablated_paths,
            ablation_type=ablation_type,
            threshold=threshold,
            grad_function=grad_function,
            answer_function=answer_function,
            circuit_out=circuit_out,
            tokens=tokens,
            alpha=alpha / edge_count, # bonferroni correction
            q_star=q_star,
        )
    # run minimality test until failure and exceeds max_edges_in_order (whichever comes second)
    ordered_test_results = {}
    has_failed = False
    for i, edge in tqdm(enumerate(edges)):
        result = test_edge(edge)
        has_failed = has_failed or result.not_minimal
        ordered_test_results[edge] = result
        if has_failed and i >= max_edges_in_order:
            break
        if i >= max_edges_in_order_without_fail:
            break
    
    return ordered_test_results

def minimality_test_edge(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    edge: Edge,
    circuit_out_inflated: BatchOutputs, 
    circuit_out_ablated_paths: BatchOutputs,
    ablation_type: AblationType,
    threshold: float,
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    use_abs: bool = True,  
    tokens: bool = False,
    circuit_out: Optional[CircuitOutputs] = None,
    alpha: float = 0.05,
    q_star: float = 0.9,
) -> MinResult:
    
    # ablate edge and run 
    prune_scores_ablated = {k: v.clone() for k, v in prune_scores.items()}
    prune_scores_ablated[edge.dest.module_name][edge.patch_idx] = 0.0
    circuit_out_ablated = next(iter(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_ablated,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
        use_abs=use_abs
    ).values()))

    # compute statistics
    n = 0
    num_edge_score_gt_ref = 0
    diffs = []
    diffs_inflated = []
    score_func = get_score_func(grad_function, answer_function)
    for batch in dataloader:
        bs = batch.clean.size(0)
        # compute frequency diff between full circuit and ablated edge is greater than inflated circuit - ablated circuit
        circ_out_ablated = circuit_out_ablated[batch.key]
        circ_out_inflated = circuit_out_inflated[batch.key]
        circ_out_ablated_paths = circuit_out_ablated_paths[batch.key]
        circ_out = circuit_out[batch.key]
        circ_out_logit_diff = score_func(circ_out, batch)
        circ_out_ablated_logit_diff = score_func(circ_out_ablated, batch)
        circ_out_inflated_logit_diff = score_func(circ_out_inflated, batch)
        circ_out_inflated_ablated_logit_diff = score_func(circ_out_ablated_paths, batch)

        circ_diff = torch.abs(circ_out_logit_diff - circ_out_ablated_logit_diff)
        circ_inflated_diff = torch.abs(circ_out_inflated_logit_diff - circ_out_inflated_ablated_logit_diff)
        num_edge_score_gt_ref += torch.sum(circ_diff > circ_inflated_diff).item()
        # log diffs
        diffs.append(circ_diff.detach().cpu())
        diffs_inflated.append(circ_inflated_diff.detach().cpu())
        n += bs
    
    p_value = binom.cdf(num_edge_score_gt_ref, n, q_star) # null is minimality p value < alpha -> not minimal 
    diffs = torch.cat(diffs)
    diffs_inflated = torch.cat(diffs_inflated)
    return MinResult(bool(p_value < alpha), p_value, num_edge_score_gt_ref, diffs, diffs_inflated)

# from auto_circuit_tests.hypo_tests.equiv_test import equiv_test, EquivResult

# class MinEquivResult(NamedTuple):
#     minimal: bool 
#     num_ablated_C_gt_M: int 
#     n: int 
#     circ_scores: torch.Tensor
#     model_scores: torch.Tensor

# # our statement should that for every edges in the circuit, ablating the edge causes non-equivlanece with 1-alpha confidence (seems reasonable)?
# def minimality_equiv_test(
#     model: PatchableModel,
#     dataloader: PromptDataLoader,
#     prune_scores: torch.Tensor | PruneScores,
#     edges: list[Edge], 

#     ablation_type: AblationType, 
#     grad_function: GradFunc,
#     answer_function: AnswerFunc,
#     threshold: Optional[float] = None,
#     edge_count: Optional[int] = None,
#     full_model_out: Dict[BatchKey, torch.Tensor]=None,
#     use_abs: bool = True,
#     alpha: float = 0.05,
#     epsilon: float = 0.1,
#     all_edges: bool = False
# ): 
#     assert (threshold is not None) ^ (edge_count is not None)
#     if threshold is None:
#         threshold = prune_scores_threshold(prune_scores, edge_count, use_abs=use_abs)
#     # TODO: finish
#     full_results = {}
#     p_value = 1.0
#     for edge in edges: # TODO: sort
#         prune_scores_ablated = {k: v.clone() for k, v in prune_scores.items()}
#         prune_scores_ablated[edge.dest.module_name][edge.patch_idx] = 0.0
#         equiv_result: EquivResult = next(iter(equiv_test(
#             model=model,
#             dataloader=dataloader,
#             prune_scores=prune_scores_ablated,
#             grad_function=grad_function,
#             answer_function=answer_function,
#             ablation_type=ablation_type,
#             thresholds=[threshold],
#             model_out=full_model_out,
#             use_abs=use_abs,
#             alpha=alpha,
#             epsilon=epsilon,
#             bayesian=True
#         ).values()))
#         p_value *= 1-equiv_result.p_value # joint probability of all edges causing non-equivlanece
#         min_equiv_result = MinEquivResult(
#             num_ablated_C_gt_M=equiv_result.num_ablated_C_gt_M,
#             n=equiv_result.n,
#             circ_scores=equiv_result.circ_scores,
#             model_scores=equiv_result.model_scores
#         )
#         full_results[edge_key(edge)] = min_equiv_result
#         if p_value > alpha and not all_edges:
#             break
#     return full_results


def plot_p_values(min_results: dict[Edge, MinResult], edge_scores: dict[Edge, torch.Tensor], alpha: float = 0.05):
    fig, ax = plt.subplots(figsize=(12, 2))
    p_values = [r.p_value for r in min_results.values()]
    neg_edge = [edge_scores[edge].cpu() < 0 for edge in min_results.keys()]
    ax.scatter(range(len(p_values), 0, -1), p_values, c=neg_edge, cmap='coolwarm')
    ax.set_xlim(len(p_values), 0)
    # plot alpha line 
    ax.axhline(y=alpha, color='g', linestyle='-')
    ax.set_title("p values for minimality test")
    return fig, ax

def plot_edge_k(min_results: dict[Edge, MinResult], edge_scores: dict[Edge, torch.Tensor], n: int, q_star: float):
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
    edge_scores: dict[Edge, torch.Tensor],
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
    diff_infl = torch.cat([r.diffs_inflated for r in min_results.values()])
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