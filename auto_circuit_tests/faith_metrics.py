# TODO: compute on train and test distribution
from functools import partial
from typing import Dict

import torch as t

from auto_circuit.types import BatchOutputs, CircuitOutputs
from auto_circuit.utils.custom_tqdm import tqdm

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, compute_scores
from auto_circuit.data import PromptPairBatch, PromptDataLoader

def compute_faith_metrics(
    dataloader: PromptDataLoader,
    model_outs: BatchOutputs,
    ablated_outs: BatchOutputs,
    circs_outs: CircuitOutputs,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
): 
    score_func = partial(compute_scores, grad_func=grad_func, answer_func=answer_func)
    faith_metrics: Dict[int, Dict[str, float]] = {}
    faith_metric_results: Dict[int, Dict[str, float]] = {}

    for n_edges, circ_outs in tqdm(list(circs_outs.items())):
        model_scores = []
        ablated_scores = []
        circ_scores = []

        n = 0
        for batch in dataloader: 
            batch: PromptPairBatch 
            model_out = model_outs[batch.key]
            ablated_out = ablated_outs[batch.key]
            circ_out = circ_outs[batch.key].to(model_out.device)

            # compute score and abs error, sum
            model_scores_b = score_func(model_out, batch, model_out) # NOTE: always 0 if div_ans_func
            ablated_scores_b = score_func(ablated_out, batch, model_out)
            circ_scores_b = score_func(circ_out, batch, model_out)

            model_scores.append(model_scores_b)
            ablated_scores.append(ablated_scores_b)
            circ_scores.append(circ_scores_b)
        
        # aggregate scores 
        model_scores = t.cat(model_scores)
        ablated_scores = t.cat(ablated_scores)
        circ_scores = t.cat(circ_scores)

        # compute abs errrors
        abs_errors = t.abs(model_scores - circ_scores)

        # compute means
        model_scores_mean = model_scores.mean()
        ablated_scores_mean = ablated_scores.mean()
        circ_scores_mean = circ_scores.mean()
        abs_error_mean = abs_errors.mean()

        # compute std dev 
        model_scores_std = model_scores.std()
        ablated_scores_std = ablated_scores.std()
        circ_scores_std = circ_scores.std()
        abs_error_std = abs_errors.std()

        # compute mean diff and farc mean diff recovered 
        mean_diff = model_scores_mean - circ_scores_mean
        frac_mean_diff_recovered = (circ_scores_mean - ablated_scores_mean) / (model_scores_mean - ablated_scores_mean)

        # store scores 
        faith_metrics[n_edges] = {
            "model_scores": model_scores.cpu().numpy().tolist(),
            "ablated_scores": ablated_scores.cpu().numpy().tolist(),
            "circ_scores": circ_scores.cpu().numpy().tolist(),
            "abs_errors": abs_errors.cpu().numpy().tolist(),
        }

        # log results
        faith_metric_results[n_edges] = {
            "model_scores_mean": model_scores_mean.item(),
            "model_scores_std": model_scores_std.item(),
            "ablated_scores_mean": ablated_scores_mean.item(),
            "ablated_scores_std": ablated_scores_std.item(),
            "circ_scores_mean": circ_scores_mean.item(),
            "circ_scores_std": circ_scores_std.item(),
            "abs_error_mean": abs_error_mean.item(),
            "abs_error_std": abs_error_std.item(),
            "mean_diff": mean_diff.item(),
            "frac_mean_diff_recovered": frac_mean_diff_recovered.item(),
        }
    return faith_metric_results, faith_metrics