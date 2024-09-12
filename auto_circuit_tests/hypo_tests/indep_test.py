from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple

import torch 
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer

from auto_circuit.data import PromptDataLoader, PromptPairBatch
from auto_circuit.prune import run_circuits
from auto_circuit.types import PruneScores, BatchOutputs, PatchType, AblationType
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.custom_tqdm import tqdm

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, get_score_func
# from auto_circuit_tests.hypo_tests.equiv_test import equiv_test



def hsic(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """(Hilbert-Schmidt Independence Criterion"""
    K_X = rbf_kernel(X, gamma=gamma)
    K_Y = rbf_kernel(Y, gamma=gamma)
    centerer = KernelCenterer()
    K_X_c = centerer.fit_transform(K_X)
    K_Y_c = centerer.fit_transform(K_Y)
    return np.trace(K_X_c @ K_Y_c)

class IndepResults(NamedTuple):
    not_indep: bool 
    p_value: float 


def independence_test(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    ablation_type: AblationType,
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    threshold: float,
    use_abs: bool,
    alpha: float = 0.05,
    B: int = 1000,
) -> IndepResults:
    # compute model out 
    m_out: BatchOutputs = {}
    for batch in dataloader:
        m_out[batch.key] = model(batch.clean)[model.out_slice]
    # construct independence scores, applying abs value if use_abs is False (to ablate negative edges and only take complement on positives)
    independence_scores = {k: v.clone() for k, v in prune_scores.items()}
    if not use_abs:
        for k in independence_scores:
            independence_scores[k][independence_scores[k] < 0] = threshold + 1
    # next, we run the complement of the circuit 
    c_comp_out = dict(next(iter(run_circuits(
        model,
        dataloader,
        prune_scores=independence_scores,
        thresholds=[threshold], # not sure that's really right here
        patch_type=PatchType.EDGE_PATCH, # Edge patch is patching the edges in the circuit (not sure why?)
        ablation_type=ablation_type,
        reverse_clean_corrupt=False, 
        use_abs=True,
    ).values())))

    # then, we compute the scores 
    score_func = get_score_func(grad_function, answer_function)
    m_scores = []
    c_comp_scores = []
    for batch in dataloader:
        m_scores.append(score_func(m_out[batch.key], batch)) # supposed to be looking at all output #TODO
        c_comp_scores.append(score_func(c_comp_out[batch.key], batch))
    m_scores = torch.cat(m_scores)[:, None].detach().cpu()
    c_comp_scores = torch.cat(c_comp_scores)[:, None].detach().cpu()
    sigma = torch.cdist(m_scores, c_comp_scores, p=2).median().item()
    m_scores = m_scores.numpy()
    c_comp_scores = c_comp_scores.numpy()

    # compute t_obs
    t_obs = hsic(m_scores, c_comp_scores, gamma=sigma)

    # then we compute the trace of the inner product of the cross product and itself (alternatively, the trace of the inner product of the covariance matrices)
    # we store that value, then for b iterations 
    t = 0
    for b in range(B):
        # permutate the model scores 
        perm_m_scores = np.random.permutation(m_scores)
        # compute the new HSIC value 
        t_i = hsic(perm_m_scores, c_comp_scores, gamma=sigma)
        # increment t with 1 if new value greater 
        t += t_obs < t_i
    # p value = t / B (higher p value -> more instances greater than t_obs -> more likely to be independent)
    p_value = t / B
    return IndepResults(not_indep=bool(p_value < alpha), p_value=p_value)


# class IndepEquivResults(NamedTuple):
#     indep: bool
#     num_ablated_C_gt_M: int 
#     n: int 
#     p_value: float
#     circ_scores: torch.Tensor
#     model_scores: torch.Tensor


# def independence_equiv_test(
#     model: PatchableModel,
#     dataloader: PromptDataLoader,
#     prune_scores: PruneScores,
#     ablation_type: AblationType,
#     grad_function: GradFunc,
#     answer_function: AnswerFunc,
#     threshold: float,
#     use_abs: bool,
#     alpha: float = 0.05,
#     epsilon: float = 0.1
# ): 
#     # run equiv test with edge pathing 
#     equiv_result = next(iter(equiv_test(
#        model=model, 
#        dataloader=dataloader,
#         prune_scores=prune_scores,
#         grad_function=grad_function,
#         answer_function=answer_function,
#         ablation_type=ablation_type,
#         patch_type=PatchType.EDGE_PATCH,
#         thresholds=[threshold],
#         use_abs=use_abs,
#         alpha=alpha,
#         epsilon=epsilon,
#         bayesian=True
#     ).values()))
#     p_value = 1 - equiv_result.p_value # probability of not being equivalent
#     indep_result = IndepEquivResults(
#         indep=equiv_result.p_value < alpha, # < alpha % chance of being equivalent -> >1-alpha % chance of being non-equivalent
#         num_ablated_C_gt_M=equiv_result.num_ablated_C_gt_M,
#         n=equiv_result.n,
#         p_value=p_value,
#         circ_scores=equiv_result.circ_scores,
#         model_scores=equiv_result.model_scores
#     )
#     return indep_result

# is fully ablated model equiv to model with ablated circuit
# def independence_equiv_test(
#     model: PatchableModel,
#     dataloader: PromptDataLoader,
#     prune_scores: PruneScores,
#     ablation_type: AblationType,
#     grad_function: GradFunc,
#     answer_function: AnswerFunc,
#     threshold: float,
#     use_abs: bool,
#     alpha: float = 0.05,
#     epsilon: float = 0.1
# ): 
#     # fully ablated model
#     ablated_out = next(iter(run_circuits(
#         model=model, 
#         dataloader=dataloader,
#         test_edge_counts=[model.n_edges],
#         prune_scores=model.new_prune_scores(),
#         patch_type=PatchType.EDGE_PATCH,
#         ablation_type=ablation_type,
#         reverse_clean_corrupt=False,
#         use_abs=use_abs,
#     ).values()))
    
#     # run equiv test with edge pathing 
#     equiv_result = next(iter(equiv_test(
#        model=model, 
#        dataloader=dataloader,
#         prune_scores=prune_scores,
#         grad_function=grad_function,
#         answer_function=answer_function,
#         ablation_type=ablation_type,
#         patch_type=PatchType.EDGE_PATCH,
#         thresholds=[threshold],
#         model_out=ablated_out,
#         use_abs=use_abs,
#         alpha=alpha,
#         epsilon=epsilon,
#         bayesian=True
#     ).values()))
#     indep_result = IndepEquivResults(
#         indep=not equiv_result.not_equiv, # < alpha % chance of being equivalent -> >1-alpha % chance of being non-equivalent
#         num_ablated_C_gt_M=equiv_result.num_ablated_C_gt_M,
#         n=equiv_result.n,
#         p_value=equiv_result.p_value,
#         circ_scores=equiv_result.circ_scores,
#         model_scores=equiv_result.model_scores
#     )
#     return indep_result