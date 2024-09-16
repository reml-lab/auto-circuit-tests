from typing import Dict, Optional, NamedTuple, Literal, Any
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

def join_values(d: Dict[Any, Dict]):
    return {k: v for sub_d in d.values() for k, v in sub_d.items()}

# ablate random edge in each path and run 
def remove_el(x: list, idx: int) -> list:
    return x[:idx] + x[idx+1:]

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


def result_to_json(result: NamedTuple): 
    return {
        k: v.tolist() if isinstance(v, torch.Tensor) else v 
        for k, v in result._asdict().items()
    }