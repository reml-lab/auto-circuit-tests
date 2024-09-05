#!/usr/bin/env python
# coding: utf-8

# In[1]:


# set cuda visible devices
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

import os
if is_notebook():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4" #"1"
    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"


# In[2]:


import torch 
torch.cuda.is_available()


# # Hypothesis Testing Automatically Discovered Circuits
# 
# Procedure: 
# - Compute prune scores (via attribution patching) 
# - Search over different thresholds to find the smallest circuit where the null hypotheis of Equivalence / Dominance cannot be rejected 
# - Prune edges from circuit that are not in paths to the output, or in the case of resample ablation cannot be reached from the input
# - Test whether each edge in the circuit is minimal 
# - Test whether the circuit is complete (by seeing if the null hypothesis on the independence test can be rejected)
# 
# 

# In[3]:


import os
from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
from itertools import product
from copy import deepcopy
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from scipy.stats import binom, beta

import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from auto_circuit.types import BatchKey, AblationType
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune_algos.activation_patching import act_patch_prune_scores
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold, flat_prune_scores

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc

from auto_circuit_tests.hypo_tests.equiv_test import (
    Side,
    equiv_test,
    sweep_search_smallest_equiv,
    plot_num_ablated_C_gt_M, 
    plot_circuit_and_model_scores,
    compute_knees, 
    plot_edge_scores_and_knees,
)
from auto_circuit_tests.hypo_tests.minimality_test import (
    minimality_test, 
    plot_p_values, 
    plot_edge_k, 
    plot_score_quantiles,
)
from auto_circuit_tests.hypo_tests.indep_test import independence_test
from auto_circuit_tests.hypo_tests.utils import (
    edges_from_mask, 
    result_to_json, 
)
from auto_circuit_tests.edge_graph import (
    SeqGraph,  
    sample_paths, 
    sample_path_random_walk_rejection, 
    SampleType,
    edge_in_path
)

from auto_circuit_tests.tasks import TASK_DICT
from auto_circuit_tests.utils import OUTPUT_DIR, RESULTS_DIR, repo_path_to_abs_path, load_cache, save_cache, save_json, load_json


# In[10]:


# config class
@dataclass 
class Config: 
    task: str = "Docstring Token Circuit" # check how many edges in component circuit (probably do all but ioi toen)
    use_abs: bool = False
    ablation_type: Union[AblationType, str] = AblationType.TOKENWISE_MEAN_CLEAN
    grad_func: Optional[Union[GradFunc, str]] = GradFunc.LOGIT
    answer_func: Optional[Union[AnswerFunc, str]] = AnswerFunc.MAX_DIFF
    ig_samples: Optional[int] = None
    layerwise: bool = False
    act_patch: bool = True
    alpha: float = 0.05
    epsilon: Optional[float] = 0.0
    q_star: float = 0.9 
    grad_func_mask: Optional[Union[GradFunc, str]] = None
    answer_func_mask: Optional[Union[AnswerFunc, str]] = None
    # clean_corrupt: Optional[str] = None #TODO: make enum
    sample_type: Union[SampleType, str] = SampleType.RANDOM_WALK
    side: Optional[Union[Side, str]] = None
    max_edges_to_test_in_order: int = 100 #TODO: change to 125
    max_edges_to_test_without_fail: int = 500 #TODO: change to 125
    max_edges_to_sample: int = 100 # TODO: change to 125
    save_cache: bool = True
    
    def __post_init__(self):
        if isinstance(self.ablation_type, str):
            self.ablation_type = AblationType[self.ablation_type.upper()]
        if isinstance(self.grad_func, str):
            self.grad_func = GradFunc[self.grad_func.upper()]
        elif self.grad_func is None:
            self.grad_func = GradFunc.LOGPROB if not self.use_abs else GradFunc.LOGIT
        if isinstance(self.answer_func, str):
            self.answer_func = AnswerFunc[self.answer_func.upper()]
        elif self.answer_func is None:
            self.answer_func = AnswerFunc.AVG_VAL if not self.use_abs else AnswerFunc.AVG_DIFF
        if self.epsilon is None:
            self.epsilon = 0.1 if self.use_abs else 0.0
        if isinstance(self.grad_func_mask, str):
            self.grad_func_mask = GradFunc[self.grad_func_mask.upper()]
        elif self.grad_func_mask is None:
            self.grad_func_mask = self.grad_func
        if isinstance(self.answer_func_mask, str):
            self.answer_func_mask = AnswerFunc[self.answer_func_mask.upper()]
        if isinstance(self.sample_type, str):
            self.sample_type = SampleType[self.sample_type.upper()]
        elif self.answer_func_mask is None:
            self.answer_func_mask = self.answer_func
        # always override clean_corrupt for now
        self.clean_corrupt = "corrupt" if self.ablation_type == AblationType.RESAMPLE else None
        if self.epsilon < 0: 
            self.side = Side.LEFT
        elif isinstance(self.side, str):
            self.side = Side[self.side.upper()]
        else: 
            self.side = Side.NONE if self.use_abs else Side.LEFT #TODO: fix this? for abs and negative epsilon?


# In[24]:


# initialize config 
conf = Config()
#get config overrides if runnign from command line
if not is_notebook():
    import sys 
    conf_dict = OmegaConf.merge(OmegaConf.structured(conf), OmegaConf.from_cli(sys.argv[1:]))
    conf = Config(**conf_dict)


# In[22]:


conf.ig_samples


# In[13]:


# handle directories
out_dir = RESULTS_DIR
out_dir.mkdir(exist_ok=True)
# should resctructure output directory to be task/ablation_type/output_func_answer_func/prune_score/experiments
task_dir = out_dir / conf.task.replace(' ', '_')
ablation_dir = task_dir / conf.ablation_type.name
out_answer_dir = ablation_dir / f"{conf.grad_func.name}_{conf.answer_func.name}"
ps_dir = out_answer_dir / (f"{conf.ig_samples}_{conf.layerwise}" if not conf.act_patch else "act_patch")
exp_dir = ps_dir / f"{conf.use_abs}_{conf.alpha}_{conf.epsilon}_{conf.q_star}"
exp_dir.mkdir(exist_ok=True, parents=True)


# In[14]:


# initialize task
task = TASK_DICT[conf.task]
task.init_task()


# # Prune Scores

# ## Activation Patching Prune Scores

# In[8]:


# load from cache if exists 
act_ps_path = out_answer_dir / "act_patch_prune_scores.pkl"
if act_ps_path.exists():
    act_prune_scores = torch.load(act_ps_path)
    act_prune_scores = {mod_name: -score for mod_name, score in act_prune_scores.items()} # negative b/c high score should imply large drop in performance
else:
    act_prune_scores = None

# if act_patch and act_patch doesn't exist, exit
if conf.act_patch and act_prune_scores is None:
    print("act_patch_prune_scores.pkl not found, exiting")
    exit()


# ##  Attribution Patching Prune Scores

# In[15]:


if not conf.act_patch:
    attr_ps_name = "attrib_patch_prune_scores"
    attr_ps_path = (ps_dir / attr_ps_name).with_suffix(".pkl")
    if (attr_ps_path).exists():
        attr_prune_scores = torch.load(attr_ps_path)
    else: 
        max_layer = max([edge.src.layer for edge in task.model.edges])

        attr_prune_scores = mask_gradient_prune_scores(
            model=task.model, 
            dataloader=task.train_loader,
            official_edges=None,
            grad_function=conf.grad_func_mask.value, 
            answer_function=conf.answer_func_mask.value, #answer_function,
            mask_val=0.0 if conf.ig_samples is None else None, 
            ablation_type=conf.ablation_type,
            integrated_grad_samples=conf.ig_samples, 
            layers=max_layer if conf.layerwise else None,
            clean_corrupt=conf.clean_corrupt,
        )
        if conf.save_cache:
            torch.save(attr_prune_scores, attr_ps_path)


# ##  Compare Activation and Attribution Patching

# In[16]:


if not conf.act_patch:
    act_prune_scores_flat = flat_prune_scores(act_prune_scores)
    attr_prune_scores_flat = flat_prune_scores(attr_prune_scores)


# ### MSE

# In[17]:


# mse and median se
if not conf.act_patch and act_prune_scores is not None:
    mse_result_name = "act_attr_mse"
    mse_result_path = (ps_dir / mse_result_name).with_suffix(".json")
    if mse_result_path.exists():
        mse_result = load_json(ps_dir, mse_result_name + '.json')
    else:
        prune_score_diffs = [
            (act_prune_scores[mod_name] - attr_prune_scores[mod_name]).flatten()
            for mod_name, _patch_mask in task.model.patch_masks.items()
        ]
        sq_error = torch.concat(prune_score_diffs).pow(2)
        median_se = sq_error.median()
        mean_se = sq_error.mean()
        mse_result = {
            "median_se": median_se.item(),
            "mean_se": mean_se.item(),
        }
        save_json(mse_result, ps_dir, mse_result_name)
    print(mse_result)


# ### Kendall Tau

# In[83]:


# # order difference 
# import itertools
# def kendallTau(A, B):
#     pairs = itertools.combinations(range(0, len(A)), 2)
#     pairs = list(pairs)
#     distance = 0

#     for x, y in tqdm(pairs):
#         a = A[x] - A[y]
#         b = B[x] - B[y]

#         # if discordant (different signs)
#         if (a * b < 0):
#             distance += 1

#     return distance

# if act_prune_scores is not None:
#     kt_results_name = "act_attr_kendall_tau"
#     kt_results_path = (ps_dir / kt_results_name).with_suffix(".json")
#     if kt_results_path.exists():
#         mse_results = load_json(ps_dir, kt_results_name + '.json')
#         dist = mse_results["distance"]
#         norm_dist = mse_results["normalized_distance"]
#     else:
#         n_elements = len(act_prune_scores_flat)

#         dist = kendallTau(act_prune_scores_flat, attr_prune_scores_flat)

#         norm_dist = dist * 2 / (n_elements * (n_elements - 1))
#         norm_dist

#         kendall_tau_result = {
#             "distance": dist,
#             "normalized_distance": norm_dist,
#         }
#         save_json(kendall_tau_result, ps_dir, kt_results_name)
#     print(kendall_tau_result)


# ### Spearman Rank Correlation

# In[18]:


if not conf.act_patch and act_prune_scores is not None:
    from scipy import stats 
    abs_corr, abs_p_value = stats.spearmanr(act_prune_scores_flat.abs().cpu(), attr_prune_scores_flat.abs().cpu())
    pos_attr_prune_scores_idx = attr_prune_scores_flat > 0
    attr_prune_scores_pos = attr_prune_scores_flat[pos_attr_prune_scores_idx]
    act_prune_scores_pos = act_prune_scores_flat[pos_attr_prune_scores_idx]
    corr, p_value = stats.spearmanr(act_prune_scores_pos.cpu(), attr_prune_scores_pos.cpu())
    print(f"abs corr: {abs_corr}, abs p-value: {abs_p_value}")
    print(f"corr: {corr}, p-value: {p_value}")

    spearman_results = {
        "abs_corr": abs_corr,
        "abs_p_value": abs_p_value,
        "corr": corr,
        "p_value": p_value,
    }
    save_json(spearman_results, ps_dir, "spearman_results")


# ### Plot Scores

# In[19]:


if not conf.act_patch and act_prune_scores is not None:
    # plot scores on x, y
    plt.scatter(act_prune_scores_flat.cpu(), attr_prune_scores_flat.cpu())
    plt.xlabel("Act Patch Scores")
    plt.ylabel("Attrib Patch Scores")
    plt.savefig(ps_dir / "act_attr_scores.png")


# In[ ]:


# check if exp_dir is empty, and exit if so (temporary for running act patch comparisions without rerunning tests)
if len(list(exp_dir.iterdir())) != 0:
    exit()


# #  Minimal Faithful Circuit According to Prune Score Ordering

# In[43]:


# set prune scores
prune_scores = act_prune_scores if conf.act_patch else attr_prune_scores


# In[44]:


model_out_train: dict[BatchKey, torch.Tensor] = {
    batch.key: task.model(batch.clean)[task.model.out_slice] 
    for batch in task.train_loader
}
model_out_test: dict[BatchKey, torch.Tensor] = {
    batch.key: task.model(batch.clean)[task.model.out_slice] 
    for batch in task.test_loader
}


# In[47]:


equiv_results, min_equiv = sweep_search_smallest_equiv(
    model=task.model, 
    dataloader=task.train_loader,
    prune_scores=prune_scores,
    grad_function=conf.grad_func, 
    answer_function=conf.answer_func,
    ablation_type=conf.ablation_type,
    use_abs=conf.use_abs,
    side=conf.side,
    alpha=conf.alpha,
    epsilon=conf.epsilon,
    model_out=model_out_train,
)
# save_json({k: result_to_json(v) for k, v in equiv_results.items()}, exp_dir, "equiv_results")


# In[30]:


equiv_test_result = equiv_test(
    model=task.model, 
    dataloader=task.test_loader,
    prune_scores=prune_scores,
    grad_function=conf.grad_func, 
    answer_function=conf.answer_func,
    ablation_type=conf.ablation_type,
    edge_counts=[min_equiv],
    use_abs=conf.use_abs,
    side=conf.side,
    alpha=conf.alpha,
    epsilon=conf.epsilon,
    model_out=model_out_test
)[min_equiv]
save_json(result_to_json(equiv_test_result), exp_dir, "equiv_test_result")


# In[31]:


threshold = prune_scores_threshold(prune_scores, min_equiv, use_abs=conf.use_abs)
edge_mask = {k: (torch.abs(v) if conf.use_abs else v) >= threshold for k, v in prune_scores.items()}
edges = edges_from_mask(task.model.srcs, task.model.dests, edge_mask, token=task.token_circuit)
save_json([edge.name for edge in edges], exp_dir, "edges")


# In[32]:


# contruct a graph from the pruned circuit, to further prune
circ_graph = SeqGraph(edges, token=task.token_circuit, attn_only=task.model.cfg.attn_only)


# In[33]:


valid_edges = [
    edge for edge in edges 
    if edge_in_path(
        edge, 
        circ_graph, 
        reach_req=conf.ablation_type == AblationType.RESAMPLE, # only works for sample b/c E[f(X)] != f(E[X])
        in_path_req=True
    )
]
min_equiv_valid_edges = len(valid_edges)
save_json([edge.name for edge in valid_edges], exp_dir, "valid_edges")
# mask out all edges not in edges to dest
# edge_score_mask = {k: ((torch.abs(v) if conf.use_abs else v) >= threshold).to(torch.int) for k, v in prune_scores.items()}
invalid_edge_score_mask = task.model.circuit_prune_scores(set(edges) - set(valid_edges)) # edges to remove
valid_edge_scores = {k: prune_scores[k] * (1-invalid_edge_score_mask[k]) for k in prune_scores.keys()}
if not sum([torch.sum(((torch.abs(v) if conf.use_abs else v) >= threshold) & (v != 0)) for v in valid_edge_scores.values()]) == len(valid_edges):
    print("Warning - valid edge scores do not match valid edges")


# In[34]:


# from elk_experiments.auto_circuit.circuit_hypotests import equiv_test
# recompute equivalence 
valid_edges_equiv_result = next(iter(equiv_test(
    task.model, 
    task.train_loader,
    valid_edge_scores,
    conf.grad_func,
    conf.answer_func,
    conf.ablation_type,
    edge_counts=[min_equiv_valid_edges],
    model_out=model_out_train,
    full_model=None,
    use_abs=conf.use_abs,
    side=conf.side,
    alpha=conf.alpha,
    epsilon=conf.epsilon,
).values()))
save_json(result_to_json(valid_edges_equiv_result), exp_dir, "valid_edges_equiv_result")


# In[42]:


# set cicuit under test
if valid_edges_equiv_result.not_equiv:
    edges_under_test = edges 
    circuit_prune_scores = prune_scores
else: 
    edges_under_test = valid_edges
    circuit_prune_scores = valid_edge_scores
edges_under_test_scores = {edge: prune_scores[edge.dest.module_name][edge.patch_idx] for edge in edges_under_test}
edges_under_test = sorted(edges_under_test_scores.keys(), key=lambda x: abs(edges_under_test_scores[x]), reverse=False)


# In[43]:


fig = draw_seq_graph(
    model=task.model,
    prune_scores=circuit_prune_scores,
    score_threshold=threshold,
    show_all_seq_pos=True,
    orientation="h",
    use_abs=conf.use_abs,
    display_ipython=is_notebook(),
    seq_labels=task.test_loader.seq_labels,
)
fig.write_image(repo_path_to_abs_path(exp_dir / "valid_edge_graph.png"))


# In[44]:


fig, ax = plot_num_ablated_C_gt_M(equiv_results, epsilon=conf.epsilon, min_equiv=min_equiv, side=conf.side)
fig.savefig(repo_path_to_abs_path(exp_dir / "num_ablated_C_gt_M.png"))


# In[45]:


fig, ax = plot_circuit_and_model_scores(equiv_results, min_equiv)
fig.savefig(repo_path_to_abs_path(exp_dir / "circuit_model_scores.png"))


# In[46]:


# plot attribution scores 
import numpy as np
edge_scores = np.flip(desc_prune_scores(prune_scores, use_abs=conf.use_abs).detach().cpu().numpy())
if not conf.use_abs:
    edge_scores = edge_scores[edge_scores > 0]
kneedle_poly, kneedle_1d = compute_knees(edge_scores)
fig, ax = plot_edge_scores_and_knees(edge_scores, kneedle_poly, kneedle_1d, min_equiv)
fig.savefig(repo_path_to_abs_path(exp_dir / "edge_scores_knees.png"))


# In[47]:


# zoom in to attribution scores
ax.set_xlim(len(edge_scores) - kneedle_poly.knee + 1,)
ax.set_yscale('linear')
ax.set_title("Edge Scores and Knees (Linear Scale)")
fig.savefig(repo_path_to_abs_path(exp_dir / "edge_scores_knees_linear.png"))
fig


# In[48]:


if min_equiv == task.model.n_edges:
    print("Circuit is entire model, cannot test minimality or independence")
    exit()


# # Minimality Test

# In[49]:


# build full grap to sample paths
graph = SeqGraph(task.model.edges, token=task.token_circuit, attn_only=task.model.cfg.attn_only)


# In[50]:


# sample paths to be used for testinga
#TODO: change to uniform sampling, but compute path counts in..hmm can i do some kind of forward propogation?
n_paths = 256
complement_edges = set(task.model.edges) - set(edges_under_test)
filtered_paths_uniform = sample_paths(graph, n_paths, complement_edges, sample_type=conf.sample_type)
assert all([any([edge in complement_edges for edge in path]) for path in filtered_paths_uniform])


# In[51]:


# run minimality test
min_test_results, min_test_sampled_results = minimality_test(
    model=task.model, 
    dataloader=task.test_loader,
    prune_scores=circuit_prune_scores,
    edges=edges_under_test, 
    edge_count=len(edges_under_test),
    ablation_type=conf.ablation_type,
    grad_function=conf.grad_func,
    answer_function=conf.answer_func,
    filtered_paths=filtered_paths_uniform,
    use_abs=conf.use_abs,
    tokens=task.token_circuit,
    alpha=conf.alpha, 
    q_star=conf.q_star, 
    max_edges_in_order=conf.max_edges_to_test_in_order,
    max_edges_in_order_without_fail=conf.max_edges_to_test_without_fail,
    max_edges_to_sample=conf.max_edges_to_sample,
)
save_json({e.name: result_to_json(r) for e, r in min_test_results.items()}, exp_dir, "min_test_results")
save_json({e.name: result_to_json(r) for e, r in min_test_sampled_results.items()}, exp_dir, "min_test_sampled_results")


# In[52]:


# minimality test on true edges
min_postfix = f"{conf.q_star}_{conf.alpha}"
min_postfix_full = f"{conf.max_edges_to_test_in_order}_{min_postfix}"
# check if already tested 
true_edges_tested = (out_answer_dir / f"min_test_true_edge_results_{min_postfix_full}.json").exists() and not is_notebook()
if not true_edges_tested:
    true_edge_prune_scores = task.model.circuit_prune_scores(task.true_edges)
    filtered_paths_true_edge = sample_paths(graph, n_paths, set(task.model.edges) - task.true_edges, sample_type=conf.sample_type)
    min_test_true_edge_results, min_test_true_edge_sampled_results = minimality_test(
        model=task.model, 
        dataloader=task.test_loader,
        prune_scores=true_edge_prune_scores,
        edges=list(task.true_edges), 
        edge_count=task.true_edge_count,
        ablation_type=conf.ablation_type,
        grad_function=conf.grad_func,
        answer_function=conf.answer_func,
        filtered_paths=filtered_paths_true_edge,
        use_abs=conf.use_abs,
        tokens=task.token_circuit,
        alpha=conf.alpha, 
        q_star=conf.q_star,
        max_edges_in_order=conf.max_edges_to_test_in_order,
        max_edges_in_order_without_fail=conf.max_edges_to_test_without_fail,
        max_edges_to_sample=conf.max_edges_to_sample,
    )
    save_json({e.name: result_to_json(r) for e, r in min_test_true_edge_results.items()}, out_answer_dir, f"min_test_true_edge_results_{min_postfix_full}")
    save_json({e.name: result_to_json(r) for e, r in min_test_true_edge_sampled_results.items()}, out_answer_dir, f"min_test_true_edge_sampled_results_{conf.max_edges_to_sample}_{min_postfix}")


# In[53]:


# plot p values``
fig, ax = plot_p_values(min_test_results, edges_under_test_scores, alpha=conf.alpha / len(edges_under_test))
fig.savefig(repo_path_to_abs_path(exp_dir / "min_test_p_values.png"))


# In[54]:


if not true_edges_tested:
    true_edge_scores = {edge: torch.tensor(1.0) for edge in task.true_edges} # don't actually know
    fig, ax = plot_p_values(min_test_true_edge_results, true_edge_scores, alpha=conf.alpha / task.true_edge_count)
    fig.savefig(repo_path_to_abs_path(score_dir / f"min_test_true_edge_p_values_{min_postfix_full}.png"))


# In[61]:


# plot frac of n 
batch_size = task.batch_size[1] if isinstance(task.batch_size, tuple) else task.batch_size
batch_count = task.batch_count[1] if isinstance(task.batch_count, tuple) else task.batch_count
fig, ax = plot_edge_k(min_test_results, edges_under_test_scores, batch_size * batch_count, q_star=conf.q_star)
fig.savefig(repo_path_to_abs_path(exp_dir / "min_test_edge_k.png"))


# In[56]:


if not true_edges_tested:
    batch_size = task.batch_size[1] if isinstance(task.batch_size, tuple) else task.batch_size
    batch_count = task.batch_count[1] if isinstance(task.batch_count, tuple) else task.batch_count
    fig, ax = plot_edge_k(min_test_true_edge_results, true_edge_scores, batch_size * batch_count, q_star=conf.q_star)
    fig.savefig(repo_path_to_abs_path(out_answer_dir / f"min_test_true_edge_k_{min_postfix_full}.png"))


# In[57]:


# plot average diff 
fig, ax = plot_score_quantiles(min_test_results, edges_under_test_scores, quantile_range=[0.00, 1.00])
fig.savefig(repo_path_to_abs_path(exp_dir / f"min_test_score_quantiles.png"))


# In[58]:


if not true_edges_tested:
    fig, ax = plot_score_quantiles(min_test_true_edge_results, true_edge_scores, quantile_range=[0.00, 1.00])
    fig.savefig(repo_path_to_abs_path(score_dir / f"min_test_true_edge_score_quantiles_{min_postfix_full}.png"))


# # Independence Test
# Test for completeness - if the circuit contains all the components required to perform the task, then the output of the complement should be independent of the original model
# 
# $H_0$: Score of complement indepedendent of score of model
# 
# Hilbert Schmdit Indepednence Criterion - non-parametric measure of independence 
# 
# - Background: (see https://jejjohnson.github.io/research_journal/appendix/similarity/hsic/)
# 
# Intuition: the trace sums along the interaction terms on each data point, which 
# we expect to be larger then other interaction terms across samples if X, and Y are 
# correlated, fewer of the perumations should be greater, our p-value will be smaller, 
# and thus we're more likely to reject the null
# 
# 
# Note: the hypothesis paper defines HCIC as  K_{x,y}K_{x,y}, but can also define it as 
# {K_x}{K_y}, b/c that that equality holds in general for Cross Covariance and Auto 
# Covariance 
# 
# The paper uses $\rho$ = median(||score(complement) - score(model)||), based on this 
# paper https://arxiv.org/pdf/1707.07269
# 
# 
# 

# In[62]:


indep_result = independence_test(
    model=task.model, 
    dataloader=task.test_loader, 
    prune_scores=circuit_prune_scores, 
    ablation_type=conf.ablation_type,
    grad_function=conf.grad_func,
    answer_function=conf.answer_func,
    threshold=threshold, 
    use_abs=conf.use_abs,
    B=1000
) 
save_json(result_to_json(indep_result), exp_dir, "indep_result")
print(indep_result)


# In[60]:


if not true_edges_tested:
    indep_true_edge_result = independence_test(
        task.model, 
        task.test_loader, 
        true_edge_prune_scores, 
        ablation_type=conf.ablation_type,
        grad_function=conf.grad_func,
        answer_function=conf.answer_func,
        threshold=1.0, 
        use_abs=True,
        alpha=conf.alpha,
        B=1000
    )
    save_json(result_to_json(indep_true_edge_result), out_answer_dir, f"indep_true_edge_result")

