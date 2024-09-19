from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
import math
from enum import Enum

import torch 
import numpy as np
from scipy.stats import binom, beta

import matplotlib.pyplot as plt

from auto_circuit.data import PromptDataLoader
from auto_circuit.prune import run_circuits
from auto_circuit.types import (
    CircuitOutputs, 
    BatchKey,
    BatchOutputs,
    PruneScores,
    PatchType, 
    AblationType,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.custom_tqdm import tqdm

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, get_score_func

class Side(Enum): 
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"

def compute_num_C_gt_M(
    circ_out: CircuitOutputs, 
    model_out: CircuitOutputs, 
    dataloader: PromptDataLoader, 
    grad_function: GradFunc,
    answer_function: AnswerFunc,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    # compute number of samples with ablated C > M
    score_func = get_score_func(grad_function, answer_function)
    num_ablated_C_gt_M = 0
    n = 0
    circ_scores = []
    model_scores = []
    for batch in dataloader:
        bs = batch.clean.size(0)
        circ_out_batch = circ_out[batch.key]
        model_out_batch = model_out[batch.key]
        circ_score = score_func(circ_out_batch, batch)
        model_score = score_func(model_out_batch, batch)
        num_ablated_C_gt_M += torch.sum(circ_score > model_score).item()
        n += bs
        circ_scores.append(circ_score)
        model_scores.append(model_score)
    return num_ablated_C_gt_M, n, torch.cat(circ_scores), torch.cat(model_scores)

def equiv_test(
    k: int, 
    n: int, 
    alpha: float = 0.05, 
    epsilon: float = 0.1, 
    null_equiv: bool=True
) -> tuple[bool, float]:
    # if null equiv, run standard two-tailed test from paper
    if null_equiv:
        theta = 1 / 2 + epsilon
        left_tail = binom.cdf(min(n-k, k), n, theta)
        right_tail = 1 - binom.cdf(max(n-k, k), n, theta)
        p_value = left_tail + right_tail
        reject_null = p_value < alpha 
    else: 
        # run two one-tailed tests (TOTS)
        # assume p = 1/2 + epsilon
        # compute probability of <= k successes given p 
        left_tail = binom.cdf(k, n, 1 / 2 + epsilon)
        # then assume p = 1/2 - epsilon
        # compute probability of >= k successes given p
        right_tail = 1 - binom.cdf(k, n, 1 / 2 - epsilon)
        reject_null = left_tail < alpha and right_tail < alpha
    return bool(reject_null), left_tail, right_tail



# def bernoulli_range_test(
#     K,
#     N,
#     eps=0.1,
#     a=[1,1],
#     alpha=0.5
# ):
#     #Inputs:
#     #  K: number of successes
#     #  N: number of trials
#     #  eps: faithfulness threshold
#     #  a: beta prior coefficients on pi
#     #  alpha: rejection threshold  
#     #Outputs: 
#     #  p(0.5-eps <= pi <= 0.5+eps | N, K, a)
#     #  p(0.5-eps <= pi <= 0.5+eps | N, K, a)<1-alpha

#     p_piK     = beta(N-K+a[0],K+a[1])
#     p_between = p_piK.cdf(0.5+eps) - p_piK.cdf(0.5-eps)
#     return(p_between<1-alpha, p_between)

class EquivResult(NamedTuple):
    num_ablated_C_gt_M: int
    n: int
    null_equiv: bool
    reject_null: bool
    left_tail: float
    right_tail: float
    circ_scores: list[float]
    model_scores: list[float]

def equiv_tests(
    model: PatchableModel, 
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    ablation_type: AblationType,
    patch_type: PatchType = PatchType.TREE_PATCH,
    edge_counts: Optional[list[int]] = None,
    thresholds: Optional[list[float]] = None,
    model_out: Optional[BatchOutputs] = None,
    circuit_outs: Optional[CircuitOutputs] = None,
    null_equiv: bool = True,
    alpha: float = 0.05,
    epsilon: float = 0.1,
) -> Dict[int, EquivResult]:

    # circuit out
    if circuit_outs is None:
        circuit_outs = run_circuits(
            model=model, 
            dataloader=dataloader,
            test_edge_counts=edge_counts,
            thresholds=thresholds,
            prune_scores=prune_scores,
            patch_type=patch_type,
            ablation_type=ablation_type,
            reverse_clean_corrupt=False,
        )
    
    # model out
    if model_out is None:
        model_out = {
            batch.key: model(batch.clean)[model.out_slice] for batch in dataloader
        }
    
    # run statitiscal tests for each edge count
    test_results = {}
    for edge_count, circuit_out in circuit_outs.items():
        num_ablated_C_gt_M, n, circ_scores, model_scores = compute_num_C_gt_M(
            circuit_out, model_out, dataloader, grad_function, answer_function
        )
        reject_nul, left_tail, right_tail = equiv_test(
            num_ablated_C_gt_M, n, alpha, epsilon, null_equiv=null_equiv
        )
        test_results[edge_count] = EquivResult(
            num_ablated_C_gt_M, 
            n, 
            null_equiv=null_equiv,
            reject_null=reject_nul,
            left_tail=left_tail,
            right_tail=right_tail,
            circ_scores=circ_scores.detach().cpu().numpy().tolist(), 
            model_scores=model_scores.detach().cpu().numpy().tolist()
        )
    return test_results




# def brute_force_equiv_test(
#     model: PatchableModel,
#     dataloader: PromptDataLoader,
#     prune_scores: PruneScores,
#     grad_function: GradFunc,
#     answer_function: AnswerFunc,
#     ablation_type: AblationType, 
#     use_abs: bool = True,
#     side: Side = Side.NONE,
#     alpha: float = 0.05,
#     epsilon: float = 0.1,
#     bayesian: bool = False,
#     model_out: Optional[Dict[BatchKey, torch.Tensor]] = None,
#     start_edge_count: int = 1,
#     reverse: bool = False,
# ): 
#     full_results = {}
#     if not reverse:
#         stop_count = model.n_edges + 1 
#         incr = 1
#     else:
#         stop_count = 0
#         incr = -1
#     edge_count_iter = tqdm(range(start_edge_count, stop_count, incr), desc="Equiv Test")
#     for edge_count in edge_count_iter:
#         equiv_result_dict = equiv_test(
#             model=model, 
#             dataloader=dataloader,
#             prune_scores=prune_scores,
#             grad_function=grad_function,
#             answer_function=answer_function,
#             ablation_type=ablation_type,
#             edge_counts=[edge_count],
#             use_abs=use_abs,
#             side=side,
#             alpha=alpha,
#             epsilon=epsilon,
#             bayesian=bayesian,
#             model_out=model_out,
#         )
#         assert len(equiv_result_dict) == 1
#         full_results.update(equiv_result_dict)
#         edge_count = next(iter(equiv_result_dict.keys()))
#         # add p value to description
#         equiv_result = equiv_result_dict[edge_count]
#         edge_count_iter.set_description(
#             f"Equiv Test: {edge_count} p={equiv_result.p_value:.3f}, k={equiv_result.num_ablated_C_gt_M}, n={equiv_result.n}")
#         if not equiv_result.not_equiv:
#             break
#     return {k: full_results[k] for k in sorted(full_results.keys())}, edge_count


# def sweep_search_smallest_equiv(
#     model: PatchableModel,
#     dataloader: PromptDataLoader,
#     prune_scores: PruneScores,
#     grad_function: GradFunc,
#     answer_function: AnswerFunc,
#     ablation_type: AblationType, 
#     use_abs: bool = True,
#     side: Side = Side.NONE,
#     alpha: float = 0.05,
#     epsilon: float = 0.1,
#     bayesian: bool = False,
#     model_out: Optional[Dict[BatchKey, torch.Tensor]] = None,
# ) -> tuple[dict[int, EquivResult], int]:
#     """Returns equiv test results and minimal equivalent number of edges."""
#     full_results = {}
#     width = 10 ** math.floor(math.log10(model.n_edges)-1)
#     interval_min = 0 
#     interval_max = model.n_edges #FIXME: if not use_abs, should only look at positive values
#     if model_out is None:
#         model_out = {batch.key: model(batch.clean)[model.out_slice] for batch in dataloader}
#     while width > 0:
#         print(f"interval: {interval_min} - {interval_max}")
#         print("width", width)
#         edge_counts = [i for i in range(interval_min, interval_max, width)]
#         edge_counts.append(interval_max)
#         test_results = equiv_test(
#             model=model, 
#             dataloader=dataloader,
#             prune_scores=prune_scores,
#             grad_function=grad_function,
#             answer_function=answer_function,
#             ablation_type=ablation_type,
#             edge_counts=edge_counts,
#             model_out=model_out,
#             full_model=None,
#             use_abs=use_abs,
#             side=side,
#             alpha=alpha,
#             epsilon=epsilon,
#             bayesian=bayesian,
#         )
#         full_results.update(test_results)
#         # find lowest interval where equivalence holds
#         equivs = [k for k, v in test_results.items() if not v.not_equiv]
#         min_equiv = min(equivs) if equivs else model.n_edges
#         # round up to width or n_edges
#         if min_equiv % width != 0:
#             min_equiv = min(min_equiv + width - min_equiv % width, model.n_edges)
#         # cases
#         new_width = width // 10
#         if min_equiv == model.n_edges:
#             interval_max = model.n_edges
#             if len(test_results) == 1:
#                 interval_min = model.n_edges
#                 new_width = 0 # exit loop
#             else:
#                 interval_min = edge_counts[-2] # get last edge count tested before full
#         else:
#             interval_max = min_equiv
#             interval_min = min_equiv - width
#         width = new_width
#     del model_out
#     full_results = {k: full_results[k] for k in sorted(full_results.keys())}
#     return full_results, interval_max
    


# def bin_search_smallest_equiv(
#     model: PatchableModel,
#     dataloader: PromptDataLoader,
#     prune_scores: PruneScores,
#     grad_function: GradFunc,
#     answer_function: AnswerFunc,
#     ablation_type: AblationType, 
#     use_abs: bool = True,
#     side: Side = Side.NONE,
#     alpha: float = 0.05,
#     epsilon: float = 0.1,
#     bayesian: bool = False,
# ):
#     edge_count_interval = [i for i in range(model.n_edges + 1)]
#     min_equiv = edge_count_interval[-1]
#     min_equiv_p_val = 0.0
#     model_out = {batch.key: model(batch.clean)[model.out_slice] for batch in dataloader}
#     while len(edge_count_interval) > 0:
#         midpoint = len(edge_count_interval) // 2
#         edge_count = edge_count_interval[midpoint]

#         num_ablated_C_gt_M, n, not_equiv, p_value = next(iter(equiv_test(
#             model=model, 
#             dataloader=dataloader,
#             prune_scores=prune_scores,
#             grad_function=grad_function,
#             answer_function=answer_function,
#             ablation_type=ablation_type,
#             edge_counts=[edge_count],
#             model_out=model_out,
#             use_abs=use_abs,
#             side=side,
#             alpha=alpha,
#             epsilon=epsilon,
#             bayesian=bayesian,
#         ).values()))

#         if not_equiv:
#             print(f"not equiv at {edge_count}, p value : {p_value}, increase edge count")
#             edge_count_interval = edge_count_interval[midpoint+1:] # more edges 
#         else:
#             min_equiv = edge_count
#             min_equiv_p_val = p_value
#             print(f"equiv at {edge_count},  p value: {p_value}, decrease edge count")
#             edge_count_interval = edge_count_interval[:midpoint] # less edges
#     del model_out
#     return min_equiv, min_equiv_p_val



def plot_num_ablated_C_gt_M(
        results: Dict[int, Any], 
        min_equiv: int, 
        epsilon: float = 0.1, 
        side: Side = Side.NONE
    ) -> Tuple[plt.Figure, plt.Axes]:
    if not -1 < epsilon < 1:
        raise ValueError("epsilon must be a float between -1 and 1 (exclusive)")
    if epsilon < 0 and side != Side.LEFT:
        raise ValueError("epsilon is negative, side must be 'left'")

    # Extract data from results
    edge_counts = list(results.keys())
    num_ablated_C_gt_Ms = [results[edge_count][0] for edge_count in edge_counts]
    ns = [results[edge_count][1] for edge_count in edge_counts]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the positions of the bars
    x = np.arange(len(edge_counts))
    width = 0.6

    # Create the bar plot for num_ablated_C_gt_M
    ax.bar(x, num_ablated_C_gt_Ms, width, label='Num Ablated C > M', color='b', alpha=0.7)

    # Create horizontal lines for N, N/2, and N/2 ± epsilon * N
    ax.plot(x, ns, label='N', color='k', linestyle='-', linewidth=2)
    if side == Side.NONE:
        ax.plot(x, [n/2 for n in ns], label='N/2', color='r', linestyle='--', linewidth=2)
        ax.plot(x, [n/2 + epsilon*n for n in ns], label=f'N/2 + {epsilon}N', color='m', linestyle=':', linewidth=2)
        ax.plot(x, [n/2 - epsilon*n for n in ns], label=f'N/2 - {epsilon}N', color='c', linestyle=':', linewidth=2)
        # Fill the area between N/2 ± epsilon * N
        ax.fill_between(x, 
                    [n/2 - epsilon*n for n in ns], 
                    [n/2 + epsilon*n for n in ns], 
                    alpha=0.2, color='y', label=f'N/2 ± {epsilon}N range')
    elif side == Side.LEFT:
        ax.plot(x, [n/2 - epsilon*n for n in ns], label=f'N/2 - {epsilon}N', color='r', linestyle=':', linewidth=2)
        # Fill the area between N/2 - epsilon * N and N
        ax.fill_between(x,
                        [n/2 - epsilon*n for n in ns],
                        [n for n in ns],
                        alpha=0.2, color='y', label=f'N/2 - {epsilon}N range')
    
    # plot vertical dotted line for min_equiv
    min_equiv_k = next((i for i, k in enumerate(edge_counts) if k == min_equiv), None)
    ax.axvline(x=min_equiv_k, color='g', linestyle='--', label=f'Minimum Equivalent ({min_equiv})')

    # Customize the plot
    ax.set_ylabel('Count')
    ax.set_xlabel('Edge Count')
    ax.set_title(f'Number of Ablated C > M')
    ax.set_xticks(x)
    ax.set_xticklabels(edge_counts, rotation='vertical', fontsize=8)
    ax.legend()

    # Add a grid for better readability
    ax.grid(True, linestyle=':', alpha=0.7)

    # Adjust y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Adjust layout and display the plot
    fig.tight_layout()
    return fig, ax

def plot_circuit_and_model_scores(test_results: Dict[int, EquivResult], min_equiv: int) -> Tuple[plt.Figure, plt.Axes]:
    # mean and std of circ_scores 
    circ_scores_mean = {k: torch.mean(v.circ_scores) for k, v in test_results.items()}
    circ_scores_std = {k: torch.std(v.circ_scores) for k, v in test_results.items()}
    # mean and std of model scores 
    model_scores_mean = {k: torch.mean(v.model_scores) for k, v in test_results.items()}
    model_scores_std = {k: torch.std(v.model_scores) for k, v in test_results.items()}

    # Convert dictionaries to lists for easier plotting
    labels = list(circ_scores_mean.keys())
    circ_means = list(circ_scores_mean.values())
    circ_stds = list(circ_scores_std.values())

    # Assuming model_scores_mean and model_scores_std are constants
    model_mean = next(iter(model_scores_mean.values())) # Get the constant model mean
    model_std = next(iter(model_scores_std.values())) # Get the constant model std

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the positions of the bars
    x = np.arange(len(labels))

    # Create the scatter plot for circuit scores
    x = np.arange(len(labels))
    ax.errorbar(x, circ_means, yerr=circ_stds, fmt='o', capsize=5, label='Circuit Scores')

    # Create horizontal lines for N, N/2, and N/2 ± epsilon * N
    ax.plot(x, [model_mean for _ in labels], label=f'Mean Score (Mean {model_mean:.2f})', color='r', linestyle='--', linewidth=2)
    ax.plot(x, [model_mean + model_std for _ in labels], label=f'Mean + STD', color='m', linestyle=':', linewidth=2)
    ax.plot(x, [model_mean - model_std for _ in labels], label=f'Mean - STD', color='c', linestyle=':', linewidth=2)

    # Fill the area between N/2 ± epsilon * N
    ax.fill_between(x, 
                    [model_mean - model_std for _ in labels], 
                    [model_mean + model_std for _ in labels], 
                    alpha=0.2, color='y', label='Mean ± STD range')

    # Create vertical lines for the minimum equivalent key
    min_equiv_idx = next((i for i, k in enumerate(test_results) if k == min_equiv), None)
    ax.axvline(x=min_equiv_idx, color='g', linestyle='--', label=f'Minimum Equivalent ({min_equiv})')

    # Customize the plot
    ax.set_ylabel('Scores')
    ax.set_xlabel('Edge Count')
    ax.set_title('Circuit Scores vs Constant Model Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical', fontsize=8)
    ax.legend()


    # Adjust layout and display the plot
    fig.tight_layout()
    return fig, ax


def compute_knees(edge_scores):
    # compute knee 
    from kneed import KneeLocator
    import numpy as np
    x = np.linspace(0, len(edge_scores), len(edge_scores))
    y = edge_scores
    kneedle_poly = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing", interp_method="polynomial")
    kneedle_1d = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing", interp_method="interp1d")
    return kneedle_poly, kneedle_1d

def plot_edge_scores_and_knees(edge_scores, kneedle_poly, kneedle_1d, min_equiv):
    fig, ax = plt.subplots()
    # plot edge scores with x labels max to 0 
    ax.plot(sorted(edge_scores, reverse=True))
    ax.set_xlim(len(edge_scores), 0)
    # log axis 
    ax.set_yscale('log')
    # plot knees
    knee_poly_edge_count = round(len(edge_scores) - kneedle_poly.knee)
    knee_1d_edge_count = round(len(edge_scores) - kneedle_1d.knee)
    ax.axvline(knee_poly_edge_count, color='r', linestyle='--', label=f"knee poly={knee_poly_edge_count}")
    ax.axvline(knee_1d_edge_count, color='g', linestyle='--', label=f"knee 1d={knee_1d_edge_count}")
    # plot min_equiv 
    ax.axvline(min_equiv, color='b', linestyle='--', label=f"min equiv={min_equiv}")
    ax.legend()
    ax.set_title("Edge Scores and Knees")
    ax.set_xlabel("Edge Count")
    ax.set_ylabel("Edge Score")
    return fig, ax