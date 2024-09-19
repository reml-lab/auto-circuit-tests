#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" #"1"


# In[2]:


from dataclasses import dataclass, field
from typing import List, Union, Tuple, Dict, Optional
import itertools

import torch as t
from omegaconf import OmegaConf

from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import AblationType, PruneScores, BatchKey, Edge
from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.prune_algos.activation_patching import compute_loss
from auto_circuit.utils.graph_utils import patch_mode, set_all_masks

from auto_circuit_tests.utils.utils import RESULTS_DIR


# In[3]:


@dataclass
class Config: 
    task: str = "Docstring Component Circuit"
    ablation_type: AblationType = AblationType.RESAMPLE
    grad_funcs: List[GradFunc] = field(default_factory=lambda: [GradFunc.LOGIT, GradFunc.LOGPROB])
    answer_funcs: List[AnswerFunc] = field(default_factory = lambda: [AnswerFunc.MAX_DIFF, AnswerFunc.AVG_VAL])
    clean_corrupt: Optional[str] = None
    edge_start: Optional[int] = None
    edge_range: Optional[int] = None

def conf_post_init(conf: Config):
    conf.clean_corrupt = "corrupt" if conf.ablation_type == AblationType.RESAMPLE else None


# In[4]:


conf = Config()
if not is_notebook():
    import sys 
    conf: Config = OmegaConf.merge(OmegaConf.structured(conf), OmegaConf.from_cli(sys.argv[1:]))
conf_post_init(conf)


# In[5]:


task_dir = RESULTS_DIR / conf.task.replace(" ", "_")
ablation_dir = task_dir / conf.ablation_type.name 


# In[6]:


task = TASK_DICT[conf.task]
# all in one batch b/c no grad
task.batch_size = task.batch_size * task.batch_count
task.batch_count = 1
task.init_task()


# In[7]:


# TODO: how do I break this up into smaller chuncks? I guesss just separate into edge ranges
# compute and store act patch scores for all combinations of grad_func and answer_func
prune_score_dict: Dict[Tuple[GradFunc, AnswerFunc], PruneScores] = {
    (grad_func, answer_func): task.model.new_prune_scores()
    for grad_func, answer_func in itertools.product(conf.grad_funcs, conf.answer_funcs)
}

full_model_score_dict: Dict[Tuple[GradFunc, AnswerFunc], PruneScores] = {
    (grad_func, answer_func): task.model.new_prune_scores()
    for grad_func, answer_func in itertools.product(conf.grad_funcs, conf.answer_funcs)
}


src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
    task.model,
    task.train_loader,
    ablation_type=conf.ablation_type,
    clean_corrupt=conf.clean_corrupt
)


# In[8]:


# sort by seq_idx, src_idx, dest.layer, dest.head
edges = sorted(task.model.edges, key=lambda edge: (edge.seq_idx, edge.src.src_idx, edge.dest.layer, edge.dest.head_idx))

# if edge_start and edge_range are set, only compute scores for those edges
if conf.edge_start is not None and conf.edge_range is not None:
    edges = edges[conf.edge_start:min(conf.edge_start + conf.edge_range, len(edges))]


# In[9]:


# compute scores on full model
with t.no_grad():
    for batch in tqdm(task.train_loader, desc="Full Model Loss"):
        logits = task.model(batch.clean)[task.model.out_slice]
        for (grad_func, answer_func) in itertools.product(conf.grad_funcs, conf.answer_funcs):
            loss = compute_loss(task.model, batch, grad_func.value, answer_func.value, logits=logits)
            for mod_name, mod in task.model.patch_masks.items():
                full_model_score_dict[(grad_func, answer_func)][mod_name] += loss.sum().item()

# compute scores for each ablated edge 
with t.no_grad():
    for edge in tqdm(edges, desc="Edge Ablation Loss"):
        edge: Edge
        set_all_masks(task.model, val=0)
        for batch in task.train_loader:
            patch_src_outs = src_outs[batch.key].clone().detach()
            with patch_mode(task.model, patch_src_outs, edges=[edge]):
                logits = task.model(batch.clean)[task.model.out_slice]
            for (grad_func, answer_func) in itertools.product(conf.grad_funcs, conf.answer_funcs):
                loss = compute_loss(task.model, batch, grad_func.value, answer_func.value, logits=logits)
                full_model_score = full_model_score_dict[(grad_func, answer_func)][edge.dest.module_name][edge.patch_idx]
                prune_score_dict[(grad_func, answer_func)][edge.dest.module_name][edge.patch_idx] = full_model_score - loss.sum().item()


# In[10]:


prune_scores_loaded = t.load(ablation_dir / f'{grad_func.name}_{answer_func.name}' / 'act_patch_prune_scores.pkl')


# In[12]:


[t.dist(prune_scores_loaded[k], prune_score_dict[(grad_func, answer_func)][k]) for k in prune_scores_loaded.keys()]


# In[10]:


# save out to directories 
file_postfix = '' if conf.edge_start is None else f'_{conf.edge_start}_{conf.edge_range}'
for (grad_func, answer_func) in itertools.product(conf.grad_funcs, conf.answer_funcs):
    score_func_name = f'{grad_func.name}_{answer_func.name}'
    ps_path = ablation_dir / score_func_name / f'act_patch_prune_scores{file_postfix}.pkl'
    print(ps_path)
    t.save(prune_score_dict[(grad_func, answer_func)], ps_path)

