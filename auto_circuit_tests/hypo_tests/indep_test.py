from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
from functools import partial

import torch 
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from scipy.spatial.distance import cdist

from auto_circuit.data import PromptDataLoader, PromptPairBatch
from auto_circuit.prune import run_circuits
from auto_circuit.types import PruneScores, BatchOutputs, CircuitOutputs, PatchType, AblationType
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.custom_tqdm import tqdm

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, compute_scores
# from auto_circuit_tests.hypo_tests.equiv_test import equiv_test



def hsic(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """(Hilbert-Schmidt Independence Criterion"""
    K_X = rbf_kernel(X, gamma=gamma)
    K_Y = rbf_kernel(Y, gamma=gamma)
    centerer = KernelCenterer()
    K_X_c = centerer.fit_transform(K_X)
    K_Y_c = centerer.fit_transform(K_Y)
    return np.trace(K_X_c @ K_Y_c)


def indep_test(X: np.ndarray, Y: np.ndarray, B: int, alpha: float) -> Tuple[bool, int, float]:
    rho = np.median(cdist(X, Y, metric='euclidean')) #torch.cdist(model_scores, comp_circuit_scores, p=2).median().item()
    gamma = 1/rho
    t_obs = hsic(X, Y, gamma=gamma)
    t = 0
    for b in range(B):
        # permutate X
        perm_X = np.random.permutation(X)
        # compute the new HSIC value 
        t_i = hsic(perm_X, Y, gamma=gamma)
        # increment t with 1 if new value greater
        t += t_obs < t_i
    p_value = t / B
    return bool(p_value < alpha), int(t), p_value


class IndepResults(NamedTuple):
    num_t_gt_t_obs: int
    B: int
    p_value: float
    reject_null: bool
    complement_model_scores: list[float]
    model_scores: list[float]


def independence_tests(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    ablation_type: AblationType,
    edge_counts: Optional[list[int]] = None,
    thresholds: Optional[list[float]] = None,
    model_out: Optional[BatchOutputs] = None,
    complement_circuit_outs: Optional[CircuitOutputs] = None,
    alpha: float = 0.05,
    B: int = 1000,
) -> Dict[int, IndepResults]:

    # compute complement outs 
    if complement_circuit_outs is None:
        complement_circuit_outs = run_circuits(
        model=model, 
        dataloader=dataloader,
        prune_scores=prune_scores,
        test_edge_counts=edge_counts,
        thresholds=thresholds,
        patch_type=PatchType.EDGE_PATCH, 
        ablation_type=ablation_type,
        reverse_clean_corrupt=True, 
    )
    # compute model out 
    if model_out is None:
        model_out: BatchOutputs = {}
        for batch in dataloader:
            model_out[batch.key] = model(batch.clean)[model.out_slice]

    # compute scores
    score_func = partial(compute_scores, grad_func=grad_func, answer_func=answer_func)
    test_results = {}
    for edge_count, comp_circuit_out in tqdm(complement_circuit_outs.items()):
        comp_circuit_scores = []
        model_scores = []
        for batch in dataloader:
            model_out_batch = model_out[batch.key]
            circ_out_batch = comp_circuit_out[batch.key]
            model_scores.append(score_func(model_out_batch, batch, model_out_batch).cpu()) 
            comp_circuit_scores.append(score_func(circ_out_batch, batch, model_out_batch).cpu())
        model_scores = torch.cat(model_scores)[:, None]
        comp_circuit_scores = torch.cat(comp_circuit_scores)[:, None]

        # run independencde test 
        reject_null, t, p_value = indep_test(
            model_scores.numpy(), comp_circuit_scores.numpy(), B=B, alpha=alpha
        )
       
        # store results
        test_results[edge_count] = IndepResults(
            num_t_gt_t_obs=t, B=B, p_value=p_value, reject_null=reject_null, 
            complement_model_scores=comp_circuit_scores.numpy().tolist(), 
            model_scores=model_scores.numpy().tolist()
        )
    return test_results



