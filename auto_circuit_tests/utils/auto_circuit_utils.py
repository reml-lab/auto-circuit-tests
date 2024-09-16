
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union
import math
from collections import defaultdict
from contextlib import ExitStack

import torch 
import torch as t
from torch.nn.functional import log_softmax
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer

from auto_circuit.types import BatchKey, CircuitOutputs, BatchOutputs, PatchWrapper
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.data import PromptDataset, PromptDataLoader, PromptPair, PromptPairBatch
from auto_circuit.types import AblationType, PatchType, PruneScores, CircuitOutputs, Edge
from auto_circuit.utils.graph_utils import patch_mode, set_mask_batch_size
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.tensor_ops import prune_scores_threshold
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.misc import module_by_name


EdgeScore = Tuple[str, str, float]

def run_fully_ablated_model(
    model: PatchableModel, 
    dataloader: PromptDataLoader,
    ablation_type: AblationType,
) -> BatchOutputs:
    """
    Runs the fully ablated model on the dataloader.
    """
    ablated_model_outs: BatchOutputs = next(iter(run_circuits(
        model=model, 
        dataloader=dataloader, 
        prune_scores=model.new_prune_scores(), 
        test_edge_counts=[model.n_edges],
        patch_type=PatchType.EDGE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=True # (all) edges are ablated with corrupt
    ).values()))
    return ablated_model_outs


def run_circuit_with_edge_ablated(
    model: PatchableModel, 
    dataloader: PromptDataLoader,
    prune_scores: PruneScores, 
    edge: Edge, 
    ablation_type: AblationType,
    threshold: float, 
    to_cpu: bool = True
) -> BatchOutputs: 
    # ablate edge  
    mask = deepcopy(prune_scores)
    mask[edge.dest.module_name][edge.patch_idx] = 0
    # run circuit
    circ_ablated_out = next(iter(run_circuits(
        model=model, 
        dataloader=dataloader,
        prune_scores=mask,
        thresholds = [threshold],
        patch_type=PatchType.TREE_PATCH, 
        ablation_type=ablation_type,
        reverse_clean_corrupt=False, 
    ).values()))
    if to_cpu:
        circ_ablated_out = {k: v.detach().cpu() for k, v in circ_ablated_out.items()}
    return circ_ablated_out


def run_circuit_with_edges_ablated(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    edges: list[Edge],
    ablation_type: AblationType,
    threshold: float,
    to_cpu: bool = True
) -> Dict[Edge, BatchOutputs]:
    edge_outs = {}
    for edge in tqdm(edges):
        edge_outs[edge] = run_circuit_with_edge_ablated(
            model=model,
            dataloader=dataloader,
            prune_scores=prune_scores,
            edge=edge,
            ablation_type=ablation_type,
            threshold=threshold,
            to_cpu=to_cpu
        )
    return edge_outs



def flat_prune_scores(prune_scores: PruneScores, per_inst: bool=False) -> t.Tensor:
    """
    Flatten the prune scores into a single, 1-dimensional tensor.

    Args:
        prune_scores: The prune scores to flatten.
        per_inst: Whether the prune scores are per instance.

    Returns:
        The flattened prune scores.
    """
    start_dim = 1 if per_inst else 0
    cat_dim = 1 if per_inst else 0
    return t.cat([ps.flatten(start_dim) for _, ps in prune_scores.items()], cat_dim)


def desc_prune_scores(prune_scores: PruneScores, per_inst: bool=False, use_abs=True) -> t.Tensor:
    """
    Flatten the prune scores into a single, 1-dimensional tensor and sort them in
    descending order.

    Args:
        prune_scores: The prune scores to flatten and sort.
        per_inst: Whether the prune scores are per instance.

    Returns:
        The flattened and sorted prune scores.
    """
    prune_scores_flat = flat_prune_scores(prune_scores, per_inst=per_inst)
    if use_abs:
        prune_scores_flat = prune_scores_flat.abs()
    return prune_scores_flat.sort(descending=True).values

def prune_scores_threshold(
    prune_scores: PruneScores | t.Tensor, edge_count: int, use_abs: bool = True
) -> t.Tensor:
    """
    Return the minimum absolute value of the top `edge_count` prune scores.
    Supports passing in a pre-sorted tensor of prune scores to avoid re-sorting.

    Args:
        prune_scores: The prune scores to threshold.
        edge_count: The number of edges that should be above the threshold.

    Returns:
        The threshold value.
    """
    if edge_count == 0:
        return t.tensor(float("inf"))  # return the maximum value so no edges are pruned

    if isinstance(prune_scores, t.Tensor):
        assert prune_scores.ndim == 1
        return prune_scores[edge_count - 1]
    else:
        return desc_prune_scores(prune_scores, use_abs=use_abs)[edge_count - 1]



def expand_patch_src_out(patch_src_out: torch.Tensor, batch_size: int):
    return patch_src_out.expand(
        patch_src_out.size(0), batch_size, patch_src_out.size(2), patch_src_out.size(3)
    )


def run_circuits(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: Optional[Union[PruneScores, Dict[BatchKey, PruneScores]]] = None,
    thresholds: Optional[List[float]] = None,
    edges: Optional[List[Edge]] = None, # compliment of circuit
    patch_type: PatchType = PatchType.EDGE_PATCH,
    ablation_type: AblationType = AblationType.RESAMPLE,
    reverse_clean_corrupt: bool = False,
    use_abs: bool = True,
    test_edge_counts: Optional[List[int]] = None,
    render_graph: bool = False,
    render_score_threshold: bool = False,
    render_file_path: Optional[str] = None,
) -> CircuitOutputs:
    """Run the model, pruning edges based on the given `prune_scores`. Runs the model
    over the given `dataloader` for each `test_edge_count`.

    Args:
        model: The model to run
        dataloader: The dataloader to use for input and patches
        test_edge_counts: The numbers of edges to prune.
        prune_scores: The scores that determine the ordering of edges for pruning
        patch_type: Whether to patch the circuit or the complement.
        ablation_type: The type of ablation to use.
        reverse_clean_corrupt: Reverse clean and corrupt (for input and patches).
        render_graph: Whether to render the graph using `draw_seq_graph`.
        render_score_threshold: Edge score threshold, if `render_graph` is `True`.
        render_file_path: Path to save the rendered graph, if `render_graph` is `True`.

    Returns:
        A dictionary mapping from the number of pruned edges to a
            [`BatchOutputs`][auto_circuit.types.BatchOutputs] object, which is a
            dictionary mapping from [`BatchKey`s][auto_circuit.types.BatchKey] to output
            tensors.
    """
    # must define method for constructing circuitt 
    if prune_scores is not None:
        assert edges is None
    else:
        assert prune_scores is None
        assert patch_type == PatchType.EDGE_PATCH #must use edge patch for patching edges 

    per_inst = isinstance(next(iter(prune_scores.values())), dict)
    circ_outs: CircuitOutputs = defaultdict(dict)
    if per_inst: 
        prune_scores_all: Dict[BatchKey, PruneScores] = prune_scores
        desc_ps_all: Dict[BatchKey: torch.Tensor] = {
            batch_key: desc_prune_scores(ps, per_inst=per_inst, use_abs=use_abs) 
            for batch_key, ps in prune_scores_all.items()
        }
    else:
        desc_ps: torch.Tensor = desc_prune_scores(prune_scores, use_abs=use_abs)
    # check if prune scores are instance specific (in which case we need to add the set_batch_size context)
  
    patch_src_outs: Optional[t.Tensor] = None
    if ablation_type.mean_over_dataset:
        patch_src_outs = src_ablations(model, dataloader, ablation_type)

    for batch_idx, batch in enumerate(batch_pbar := tqdm(dataloader)):
        batch_pbar.set_description_str(f"Pruning Batch {batch_idx}", refresh=True)
        if (patch_type == PatchType.TREE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.EDGE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.clean
            if not ablation_type.mean_over_dataset:
                patch_src_outs = src_ablations(model, batch.corrupt, ablation_type)
        elif (patch_type == PatchType.EDGE_PATCH and not reverse_clean_corrupt) or (
            patch_type == PatchType.TREE_PATCH and reverse_clean_corrupt
        ):
            batch_input = batch.corrupt
            if not ablation_type.mean_over_dataset:
                patch_src_outs = src_ablations(model, batch.clean, ablation_type)
        else:
            raise NotImplementedError

        if per_inst:
            prune_scores = prune_scores_all[batch.key]
            desc_ps = desc_ps_all[batch.key]

        if test_edge_counts is not None:
            assert per_inst is False # TODO: support
            thresholds = [prune_scores_threshold(desc_ps, edge_count, use_abs=use_abs)
                          for edge_count in test_edge_counts]
        else: 
            assert thresholds is not None
        
        assert patch_src_outs is not None
        with ExitStack() as stack:
            stack.enter_context(patch_mode(model, patch_src_outs, edges=edges))
            if per_inst:
                stack.enter_context(set_mask_batch_size(model, batch_input.size(0)))
            for threshold in tqdm(thresholds):
                if prune_scores is not None:
                    # When prune_scores are tied we can't prune exactly edge_count edges
                    patch_edge_count = 0
                    for mod_name, patch_mask in prune_scores.items():
                        dest = module_by_name(model, mod_name)
                        assert isinstance(dest, PatchWrapper)
                        assert dest.is_dest and dest.patch_mask is not None
                        if patch_type == PatchType.EDGE_PATCH:
                            dest.patch_mask.data = ((patch_mask.abs() if use_abs else patch_mask) >= threshold).float()
                            patch_edge_count += dest.patch_mask.int().sum().item()
                        else:
                            assert patch_type == PatchType.TREE_PATCH
                            dest.patch_mask.data = ((patch_mask.abs() if use_abs else patch_mask) < threshold).float()
                            patch_edge_count += (1 - dest.patch_mask.int()).sum().item()
                else: # edges is not None
                    assert edges is not None
                with t.inference_mode():
                    model_output = model(batch_input)[model.out_slice]
                circ_outs[patch_edge_count][batch.key] = model_output.detach().clone()
            if render_graph:
                draw_seq_graph(
                    model=model,
                    score_threshold=render_score_threshold,
                    show_all_seq_pos=False,
                    seq_labels=dataloader.seq_labels,
                    file_path=render_file_path,
                )
    del patch_src_outs
    return circ_outs

def load_tf_model(model_name: str):
    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True
    )
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model




