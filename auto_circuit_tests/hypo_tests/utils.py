from typing import Dict, Optional, NamedTuple, Literal
import torch
import numpy as np
from auto_circuit.types import SrcNode, DestNode, Edge
from auto_circuit.data import PromptDataLoader, PromptPairBatch
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
from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, get_score_func





def compute_frac_perf_recovered(
    model: PatchableModel, 
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    ablation_type: AblationType,
    edge_count: Optional[int] = None,
    threshold: Optional[float] = None,
    full_model_outs: Optional[BatchOutputs] = None, 
    ablated_model_outs: Optional[BatchOutputs] = None,
    metric: Literal["mse", "mae", "diff"] = "mse",
    diff_in_expected: bool = False,
) -> float:
    """
    Tests whether the fraction of full model performance (relative to fully ablated model 
    baseline) recovered by the circuit (defined by prune scores and 
    (thresholds/edge count) is >= 1-epsilon with p > 1-alpha.

    We define fraction of recovered performance as: 
    l = 1 - (E[metric(M, C)] / E[(metric(M, A))]). 
    Note that l is in [-inf, 1], and in practice we expect l to be in [0, 1] 
    (b/c the circuit will be a better approximation of the model). 

    We use a one-sided t-test with
    H0: l < 1-epsilon
    H1: l >= 1-epsilon
"""
    # compute full model out 
    if full_model_outs is None:
        full_model_outs: BatchOutputs = {
            batch.key: model(batch.clean)[model.out_slice] 
            for batch in dataloader
        }
    # compute ablated model out 
    if ablated_model_outs is None:
       ablated_model_outs: BatchOutputs = run_fully_ablated_model(
            model=model,
            dataloader=dataloader,
            ablation_type=ablation_type,
        )
    # compute circuit out
    circuit_outs: BatchOutputs = next(iter(run_circuits(
        model=model, 
        dataloader=dataloader, 
        prune_scores=prune_scores, 
        test_edge_counts=[edge_count] if edge_count is not None else None,
        thresholds=[threshold] if threshold is not None else None,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False, 
    ).values()))

    # TODO: could also do 1 - E_C - E_M / E_A - E_M

    # compute mean metric between full model and ablated model
    score_func = get_score_func(grad_func, answer_func)
    p = 2 if metric == "mse" else 1
    if diff_in_expected:
        model_scores = 0 
        ablated_scores = 0
        circuit_scores = 0
    else: 
        ablated_loss = 0 
        circuit_loss = 0
    n = 0
    for batch in dataloader:
        batch: PromptPairBatch
        full_model_out = full_model_outs[batch.key]
        ablated_model_out = ablated_model_outs[batch.key]
        circuit_out = circuit_outs[batch.key]
        
        # compute scores and loss
        full_model_score = score_func(full_model_out, batch)
        ablated_model_score = score_func(ablated_model_out, batch)
        circuit_score = score_func(circuit_out, batch)
        if diff_in_expected:
            model_scores += full_model_score.sum() 
            ablated_scores += ablated_model_score.sum() 
            circuit_scores += circuit_score.sum()
        else:
            print("p", p)
            ablated_loss += torch.dist(full_model_score, ablated_model_score, p=p)
            circuit_loss += torch.dist(full_model_score, circuit_score, p=p)
        n += full_model_out.size(0)
    if diff_in_expected:# return 1 - (E[score(M)] - E[score(C)] / E[score(M)] - E[score(A)])
        model_scores /= n
        ablated_scores /= n
        circuit_scores /= n
        if metric == "diff":
            frac_perf_recovered = 1 - (model_scores - circuit_scores) / (model_scores - ablated_scores)
        else:
            frac_perf_recovered = 1 - torch.dist(model_scores, circuit_scores, p=p) / torch.dist(model_scores, ablated_scores, p=p)
        return frac_perf_recovered, circuit_scores, ablated_scores, model_scores
    else: # return 1 - (E[metric(M, C)] / E[(metric(M, A))])
        ablated_loss /= n
        circuit_loss /= n
        return 1 - (circuit_loss / ablated_loss), circuit_loss, ablated_loss, 0

def edges_from_mask(srcs: set[SrcNode], dests: set[DestNode], mask: Dict[str, torch.Tensor], token: bool=False) -> list[Edge]:
    #TODO: fix for SAEs
    SRC_IDX_TO_NODE = {src.src_idx: src for src in srcs}
    DEST_MOD_AND_HEAD_TO_NODE = {(dest.module_name, dest.head_idx): dest for dest in dests}
    edges = []
    for mod_name, mask in mask.items():
        for idx in mask.nonzero():
            # src idx 
            if len(idx) == 1:
                assert not token
                src_idx = idx.item() 
                dest_idx = None
                seq_idx = None
            elif len(idx) == 2 and not token: 
                src_idx = idx[1].item()
                dest_idx = idx[0].item()
                seq_idx = None
            elif len(idx) == 2 and token:
                src_idx = idx[1].item()
                dest_idx = None
                seq_idx = idx[0].item()
            else: 
                assert token and len(idx) == 3
                src_idx = idx[2].item()
                dest_idx = idx[1].item()
                seq_idx = idx[0].item()
            dest_node = DEST_MOD_AND_HEAD_TO_NODE[(mod_name, dest_idx)]
            src_node = SRC_IDX_TO_NODE[src_idx]
            edges.append(Edge(src=src_node, dest=dest_node, seq_idx=seq_idx))
    return edges


def set_score(edge: Edge, scores, value, batch_idx: Optional[int] = None, tokens: bool = False):
    idx = edge.patch_idx
    # remove nones
    idx = tuple(filter(lambda x: x is not None, idx))
    # if idx[0] is None:
    #     idx = idx[1:]
    if batch_idx is not None:
        idx = (batch_idx,) + idx
    scores[edge.dest.module_name][idx] = value
    return scores

def result_to_json(result: NamedTuple): 
    return {
        k: v.tolist() if isinstance(v, torch.Tensor) else v 
        for k, v in result._asdict().items()
    }