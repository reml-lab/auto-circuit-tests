
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

from cupbearer.data import MixedData

from auto_circuit.types import BatchKey, CircuitOutputs, Measurements, PatchWrapper
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



def make_prompt_dataset(data, effect_tokens, vocab_size, device='cpu'):
    clean_prompts = [x[0] for x in data]
    answers = [effect_tokens] * len(clean_prompts)
    wrong_answers = [list(set(range(vocab_size)) - set(answer)) for answer in answers]
    
    # put into torch tensors
    clean_prompts = torch.stack(clean_prompts, dim=0)
    corrupt_prompts = torch.stack([torch.zeros_like(clean_prompts[0], dtype=int)] * len(clean_prompts), dim=0)
    answers = [torch.tensor(answer, dtype=int) for answer in answers]
    wrong_answers= [torch.tensor(answer, dtype=int) for answer in wrong_answers]

    return PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers)

def make_mixed_prompt_dataloader(dataset: MixedData, effect_tokens, model, batch_size, device='cpu'):
    normal_dataset = make_prompt_dataset(dataset.normal_data, effect_tokens, model.tokenizer.vocab_size, device=device)
    anomalous_dataset = make_prompt_dataset(dataset.anomalous_data, effect_tokens, model.tokenizer.vocab_size, device=device)
    prompt_dataset = MixedData(normal_dataset, anomalous_dataset, dataset.normal_weight, dataset.return_anomaly_labels)
    seq_len = normal_dataset.clean_prompts.size(1)
    dataloader = PromptDataLoader(
        prompt_dataset,
        seq_len=seq_len,
        diverge_idx=0,
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn
    )
    return dataloader

def make_prompt_data_loader(dataset, effect_tokens, model, batch_size):
    prompt_dataset = make_prompt_dataset(dataset, effect_tokens, model.tokenizer.vocab_size)
    seq_len = prompt_dataset.clean_prompts.size(1)
    dataloader = PromptDataLoader(
        prompt_dataset,
        seq_len=seq_len,
        diverge_idx=0,
        batch_size=batch_size,
        shuffle=False
    )
    return dataloader



def collate_fn(batch: List[Tuple[PromptPair, int]]) -> Tuple[PromptPairBatch, torch.Tensor]:
    clean = torch.stack([p.clean for p, y in batch])
    corrupt = torch.stack([p.corrupt for p, y in batch])
    labels = torch.tensor([y for p, y in batch], dtype=torch.int)
    if all([p.answers.shape == batch[0][0].answers.shape for p, y in batch]):
        answers = torch.stack([p.answers for p, y in batch])
    else:  # Sometimes each prompt has a different number of answers
        answers = [p.answers for p, y in batch]
    if all([p.wrong_answers.shape == batch[0][0].wrong_answers.shape for p, y in batch]):
        wrong_answers = torch.stack([p.wrong_answers for p, y in batch])
    else:  # Sometimes each prompt has a different number of wrong answers
        wrong_answers = [p.wrong_answers for p, y in batch]
    key = hash((str(clean.tolist()), str(corrupt.tolist())))

    diverge_idxs = (~(clean == corrupt)).int().argmax(dim=1)
    batch_dvrg_idx: int = int(diverge_idxs.min().item())
    return PromptPairBatch(key, batch_dvrg_idx, clean, corrupt, answers, wrong_answers), labels


def sorted_scores(scores: PruneScores, model: PatchableModel) -> List[EdgeScore]:
    sorted_srcs = sorted(model.srcs, key=lambda x: x.src_idx)
    sorted_dests_by_module = {mod.module_name: sorted([dest for dest in model.dests if dest.module_name == mod.module_name], key=lambda x: x.head_idx) for mod in model.dest_wrappers}
    
    score_thruple = []
    for mod_name, score in scores.items():
        if score.ndim == 1:
            score = score.unsqueeze(0)
        for i in range(score.shape[0]):
            dest_name = sorted_dests_by_module[mod_name][i].name
            for j in range(score.shape[1]):
                src_name = sorted_srcs[j].name
                score_thruple.append((src_name, dest_name, score[i, j]))
    score_thruple.sort(key=lambda x: abs(x[2]), reverse=True)
    return score_thruple


def log_answer_dist(logits, answers: torch.Tensor):
    probs = logits.softmax(dim=-1)
    answer_prob = torch.gather(probs, 1, answers).sum(dim=-1)
    answer_dist = torch.stack([answer_prob, 1 - answer_prob], dim=1)
    answer_log_dist = answer_dist.log()
    return answer_log_dist


def measure_kl_div(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    compare_to_clean: bool = True,
    over_vals: bool = False, 
) -> List[Tuple[int, float, torch.Tensor]]:
    """
    Average KL divergence between the full model and the circuits.

    Args:
        model: The model on which `circuit_outs` was calculated.
        dataloader: The dataloader on which the `circuit_outs` was calculated.
        circuit_outs: The outputs of the ablated model for each circuit size.
        compare_to_clean: Whether to compare the circuit output to the full model on the
            clean (`True`) or corrupt (`False`) prompt.
        over_vals: Whether to take KL over [answer, wrong_answer] or entire token distribution

    Returns:
        A list of tuples, where the first element is the number of edges pruned and the
            second element is the average KL divergence for that number of edges.
    """
    circuit_kl_divs: Measurements = []
    default_logprobs: Dict[BatchKey, t.Tensor] = {}
    with t.inference_mode():
        for batch in dataloader:
            default_batch = batch.clean if compare_to_clean else batch.corrupt
            logits = model(default_batch)[model.out_slice]
            default_logprobs[batch.key] = log_softmax(logits, dim=-1) if not over_vals else log_answer_dist(logits, batch.answers)

    for edge_count, circuit_out in (pruned_out_pbar := tqdm(circuit_outs.items())):
        pruned_out_pbar.set_description_str(f"KL Div for {edge_count} edges")
        circuit_logprob_list: List[t.Tensor] = []
        default_logprob_list: List[t.Tensor] = []
        for batch in dataloader:
            circuit_logprob_list.append(log_softmax(circuit_out[batch.key], dim=-1) if not over_vals else log_answer_dist(circuit_out[batch.key], batch.answers))
            default_logprob_list.append(default_logprobs[batch.key])
        
        input_logprobs = t.cat(circuit_logprob_list)
        target_logprobs = t.cat(default_logprob_list)
        n_batch = math.prod(input_logprobs.shape[:-1])
        kl_instance = torch.nn.functional.kl_div( 
            input_logprobs,
            target_logprobs,
            reduction="none",
            log_target=True,
        )
        kl_instance = kl_instance.sum(dim=-1) # sum over "clases"
        kl = kl_instance.sum() / n_batch
        # kl = multibatch_kl_div(input_logprobs, target_logprobs)
       

        # Numerical errors can cause tiny negative values in KL divergence
        circuit_kl_divs.append((edge_count, max(kl.item(), 0), torch.clip(kl_instance, 0, None).tolist()))
    return circuit_kl_divs


def plot_attribution_and_kl_div(
        model: PatchableModel,
        attribution_scores, 
        kl_divs: List[Tuple[int, float, List[float]]],
        knee, 
        title, 
        plot_interval=False,
        kl_divs_anom=None,
):
    # plot attribution scores and kl_divs
    fig, ax1 = plt.subplots()

    lines = []
    # plot attribution scores
    color = 'tab:red'
    ax1.set_xlabel('edge')
    ax1.set_ylabel('attribution score', color=color)
    attrib_line = ax1.plot(attribution_scores, color=color)
    lines.append((attrib_line[0], 'attrib'))
    ax1.tick_params(axis='y', labelcolor=color)

    # plot kl_divs
    xvals, y_vals, individual_kl_divs = zip(*kl_divs)
    percentile_95 = [np.percentile(kl, 95) for kl in individual_kl_divs]
    percentile_5 = [np.percentile(kl, 5) for kl in individual_kl_divs]
    xvals = [model.n_edges - x for x in xvals]
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('kl_div', color=color)
    
    kl_line = ax2.plot(xvals, y_vals, color=color)
    if plot_interval:
        ax2.fill_between(xvals, percentile_5, percentile_95, color=color, alpha=0.2)
    lines.append((kl_line[0], 'kl_div'))
    ax2.tick_params(axis='y', labelcolor=color)

    # plot anomalous kl_divs if present 
    if kl_divs_anom:
        xvals, y_vals, individual_kl_divs = zip(*kl_divs_anom)
        percentile_95 = [np.percentile(kl, 95) for kl in individual_kl_divs]
        percentile_5 = [np.percentile(kl, 5) for kl in individual_kl_divs]
        xvals = [model.n_edges - x for x in xvals]
        color = 'tab:green'
        kl_anom_line = ax2.plot(xvals, y_vals, color=color)
        if plot_interval:
            ax2.fill_between(xvals, percentile_5, percentile_95, color=color, alpha=0.2)
        lines.append((kl_anom_line[0], 'kl_div_anom'))

    # plot knee 
    knee_line = ax1.axvline(x=knee, color='grey', linestyle=':', label='knee')
    lines.append((knee_line, 'knee'))

    # legend
    lines, labels = zip(*lines)
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.15, 0.85))

    # title 
    plt.title(f"Attribution scores and KLDivs {title}")


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
