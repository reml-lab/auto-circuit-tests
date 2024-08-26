from typing import Dict, Optional, NamedTuple
import torch
from auto_circuit.types import SrcNode, DestNode, Edge


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


def get_edge_idx(edge: Edge, tokens=False):
    # TODO: make backwards compatible
    if edge.dest.name == "Resid End":
        idx = (edge.src.src_idx,)
    elif edge.dest.name.startswith("MLP"):
        idx = (edge.src.src_idx,)
    else:
        idx = (edge.dest.head_idx, edge.src.src_idx)
    if tokens:
        idx = (edge.seq_idx,) + idx
    return idx


def set_score(edge: Edge, scores, value, batch_idx: Optional[int] = None, tokens: bool = False):
    idx = get_edge_idx(edge, tokens=tokens)
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