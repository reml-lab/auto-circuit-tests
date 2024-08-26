from collections import defaultdict, namedtuple, OrderedDict
from typing import Tuple, Optional, Literal
import random
from enum import Enum
from dataclasses import dataclass, field

import networkx as nx
import matplotlib.pyplot as plt

from auto_circuit.types import SrcNode, DestNode, Edge, Node
from auto_circuit.utils.custom_tqdm import tqdm

class NodeType(Enum):
    Q = 0 
    K = 1 
    V = 2
    ATTN = 3
    MLP = 4
    RESID_START = 5 
    RESID_END = 6

NodeIdx = Tuple[int, int] # layer, head
SeqNodeKey = Tuple[int, int, int, NodeType] # layer, head_idx, seq_idxs, node_type


def node_name_to_type(node_name: str) -> NodeType:
    if node_name.endswith("Q"):
        return NodeType.Q
    if node_name.endswith("K"):
        return NodeType.K
    if node_name.endswith("V"):
        return NodeType.V
    if node_name.startswith("A"):
        return NodeType.ATTN
    if node_name.startswith("MLP"):
        return NodeType.MLP
    if node_name.startswith("Resid Start"):
        return NodeType.RESID_START
    if node_name.startswith("Resid End"):
        return NodeType.RESID_END
    raise ValueError(f"Invalid node name {node_name}")


def valid_node(
    layer: int, 
    node_type: NodeType, 
    seq_idx: int, 
    max_layer: int, 
    last_seq_idx: int, 
    attn_only: bool=False
) -> bool:
    before_last = ((layer < max_layer - 2) or (attn_only and layer < max_layer - 1)) 
    if before_last: 
        return True  
    if seq_idx == last_seq_idx: # last sequence index
        return True 
    if node_type in (NodeType.K, NodeType.V): # is k or value
        return True
    return False

def dest_to_src_node_type(node_type: NodeType) -> NodeType:
    if node_type in (NodeType.Q, NodeType.K, NodeType.V):
        return NodeType.ATTN
    else: 
        assert node_type in (NodeType.ATTN, NodeType.MLP)
        return node_type


@dataclass
class SeqNode:
    name: str
    layer: int 
    head_idx: int
    seq_idx: int
    node_type: NodeType
    is_src: bool 
    path_count: int 
    reachable: bool
    in_edges: list["SeqEdge"] = field(default_factory=list)
    out_edges: list["SeqEdge"] = field(default_factory=list)

    @classmethod
    def from_node(cls, node: Node, seq_idx: int, path_count: int=0, reachable: bool=False) -> "SeqNode":
        return cls(
            name=node.name,
            layer=node.layer,
            head_idx=node.head_idx,
            seq_idx=seq_idx,
            is_src=isinstance(node, SrcNode),
            node_type=node_name_to_type(node.name),
            path_count=path_count,
            reachable=reachable,
        )

def get_seq_node_key(node: SeqNode) -> SeqNodeKey:
    return (node.layer, node.head_idx, node.seq_idx, node_name_to_type(node.name))

def node_to_seq_node_key(node: Node, seq_idx: int) -> SeqNodeKey:
    return (node.layer, node.head_idx, seq_idx, node_name_to_type(node.name))

def dest_node_to_src_node_key(node: SeqNode):
    return (node.layer, node.head_idx, node.seq_idx, dest_to_src_node_type(node.node_type))

@dataclass 
class SeqEdge:
    head: SeqNode
    tail: SeqNode
    edge: Optional[Edge] = None


class SeqGraph():
    """
    Same as graph created from patchable model, but with 
    edge from dest nodes to src nodes (including cross-seq attention edges)

    Automatically removed edges which cannot reach output at last_seq_idx
    """

    def __init__(
        self, 
        edges: list[Edge],
        token: bool = True,
        attn_only: bool = False
    ):  
        # used for filtering edges
        self.last_seq_idx = max([edge.seq_idx for edge in edges]) if token else None
        self.max_layer = max([edge.dest.layer for edge in edges])
        self.token = token
        self.attn_only = attn_only

        # create srcs and dests, and edges between them
        self.seq_srcs: dict[SeqNodeKey, SeqNode] = dict()
        self.seq_dests: dict[SeqNodeKey, SeqNode] = dict()
        self.seq_nodes: list[SeqNode] = []
        self._build_graph(edges) #inits seq_srcs, seq_dests
        self._sort_nodes() # inits seq_nodes
        self._compute_path_counts() # inits path_counts on seq_nodes, and also valid edges
        self._compute_reachable()

        
    def _build_graph(self, edges: list[Edge]):
        # initialize seq srcs and dests, adding (standard) edges between src and dest
        # hmmm - seems like we're not getting all the attention connections
        for edge in edges:
            src_node_key = node_to_seq_node_key(edge.src, edge.seq_idx)
            dest_node_key = node_to_seq_node_key(edge.dest, edge.seq_idx)
            # if dest is not valid (can't reach output)
            if not self.valid_node(edge.dest.layer, dest_node_key[-1], edge.seq_idx):
                continue
            if src_node_key not in self.seq_srcs:
                self.seq_srcs[src_node_key] = SeqNode.from_node(edge.src, edge.seq_idx)
            if dest_node_key not in self.seq_dests:
                self.seq_dests[dest_node_key] = SeqNode.from_node(edge.dest, edge.seq_idx)
            seq_src = self.seq_srcs[src_node_key]
            seq_dest = self.seq_dests[dest_node_key]
            seq_edge = SeqEdge(head=seq_src, tail=seq_dest, edge=edge)
            seq_src.out_edges.append(seq_edge)
            seq_dest.in_edges.append(seq_edge)
        
        # add edges between dests and srcs 
        # required for attention edges from k,v to attention head in all subsequent seq_idxs
        attn_srcs: dict[NodeIdx, list[SeqNode]] = defaultdict(list)
        for seq_src in self.seq_srcs.values():
            if seq_src.node_type == NodeType.ATTN:
                attn_srcs[(seq_src.layer, seq_src.head_idx)].append(seq_src)
        for dest in self.seq_dests.values():
            if dest.node_type == NodeType.RESID_END:
                pass
            elif dest.node_type in (NodeType.K, NodeType.V) and self.token:
                # add attention head at all subsequent seq_idxs
                seq_srcs_to_add = [src for src in attn_srcs[(dest.layer, dest.head_idx)] 
                                   if src.seq_idx >= dest.seq_idx] # causal attention
                for seq_src in seq_srcs_to_add:
                    out_edge = SeqEdge(head=dest, tail=seq_src)
                    dest.out_edges.append(out_edge)
                    seq_src.in_edges.append(out_edge)
            else:
                # add src with same layer and head_idx
                src_idx = dest_node_to_src_node_key(dest)
                if src_idx in self.seq_srcs:
                    seq_src = self.seq_srcs[dest_node_to_src_node_key(dest)]
                    out_edge = SeqEdge(head=dest, tail=seq_src)
                    dest.out_edges.append(out_edge)
                    seq_src.in_edges.append(out_edge)
    
    def valid_node(self, layer: int, node_type: NodeType, seq_idx: int) -> bool:
        return valid_node(layer, node_type, seq_idx, self.max_layer, self.last_seq_idx, attn_only=self.attn_only)
    
    def _sort_nodes(self):
        combined_nodes: list[SeqNode] = [*self.seq_srcs.values(), *self.seq_dests.values()]
        self.seq_nodes = sorted(combined_nodes, key=lambda node: (
            node.layer, node.is_src, node.seq_idx, node.head_idx
        ))

    def _compute_path_counts(self): 
        for seq_node in reversed(self.seq_nodes):
            if seq_node.node_type == NodeType.RESID_END and seq_node.seq_idx == self.last_seq_idx:
                seq_node.path_count = 1
                continue
            if len(seq_node.out_edges) == 0:
                seq_node.path_count = 0
                continue
            seq_node.path_count = sum([edge.tail.path_count for edge in seq_node.out_edges]) # sum of children path counts

    def _compute_reachable(self):
        for seq_node in self.seq_nodes:
            if seq_node.node_type == NodeType.RESID_START:
                seq_node.reachable = True
                continue
            seq_node.reachable = any([in_edge.head.reachable for in_edge in seq_node.in_edges])


def edge_in_path(edge: Edge, seq_graph: SeqGraph, in_path_req=True, reach_req=True) -> bool:
    # check if edge src and dest are in seq_graph
    src_idx = node_to_seq_node_key(edge.src, edge.seq_idx)
    dest_idx = node_to_seq_node_key(edge.dest, edge.seq_idx)
    if src_idx not in seq_graph.seq_srcs or dest_idx not in seq_graph.seq_dests:
        return False
    # look up edge src, see if reachable 
    edge_src = seq_graph.seq_srcs[src_idx]
    if reach_req and not edge_src.reachable:
        return False
    # look up edge dest, see if path_counts > 0s
    edge_dest = seq_graph.seq_dests[dest_idx]
    if in_path_req and edge_dest.path_count == 0:
        return False
    return True


# goal is to return equivalent of path counts, but only count paths that have at least one edge in provided edges 
PathCounts = dict[Tuple[SeqNodeKey, bool], int]
def get_edge_path_counts(
    edges: list[Edge],
    seq_graph: SeqGraph,
) -> PathCounts:
    
    # create path counts default dict (key is SeqNodeKey and is_src)
    edge_path_counts: PathCounts = defaultdict(int)
    
    # filter edges for edges in path 
    edges = [edge for edge in edges if edge_in_path(edge, seq_graph, in_path_req=True, reach_req=True)]
    if len(edges) == 0:
        raise ValueError("No edges that lay in path from src to dest")
    for edge in edges: 
        edge_path_counts[(node_to_seq_node_key(edge.src, edge.seq_idx), True)] = 1
    

    max_layer = max([edge.src.layer for edge in edges])

    # get seq_nodes to process
    seq_nodes_to_process: list[SeqNode] = []
    for node in seq_graph.seq_nodes: # sorted in order of layer
        if node.layer > max_layer:
            break
        seq_nodes_to_process.append(node)

    # get path counts from child path counts
    for seq_node in tqdm(reversed(seq_nodes_to_process)):
        edge_path_counts[(get_seq_node_key(seq_node), seq_node.is_src)] = max( # take max b/c already set to 1 if has edge
            sum(
                [edge_path_counts[(get_seq_node_key(edge.tail), edge.tail.is_src)] 
                 for edge in seq_node.out_edges]
            ),
            edge_path_counts[(get_seq_node_key(seq_node), seq_node.is_src)]
        )
    return dict(edge_path_counts)

class SampleType(Enum):
    UNIFORM = 0
    RANDOM_WALK = 1

def _sample_edge_idx(
    seq_nodes: list[SeqNode], 
    edge_path_counts: Optional[PathCounts]=None,
    is_src: Optional[bool]=None,
    sample_type: SampleType=SampleType.UNIFORM
) -> Tuple[int, int]:
    if edge_path_counts: 
        path_counts = [edge_path_counts[(get_seq_node_key(node), is_src)] for node in seq_nodes]
    else: 
        path_counts = [node.path_count for node in seq_nodes]
    
    if sample_type == SampleType.RANDOM_WALK: # not proportional to path counts)
        path_counts = [int(path_count > 0) for path_count in path_counts]
    
    path_count_sum = sum(path_counts)
    probs = [path_count / path_count_sum for path_count in path_counts]

    idx = random.choices(range(len(seq_nodes)), weights=probs)[0] 
    return idx, path_counts[idx]

def _node_edge_in_edges(node: SeqNode, edges: set[Edge]) -> bool:
    return any(edge.edge in edges for edge in node.out_edges)


# sample path
def sample_path(
    seq_graph: SeqGraph, 
    edge_path_counts: Optional[PathCounts]=None, 
    edges: Optional[set[Edge]]=None, 
    sample_type: SampleType=SampleType.UNIFORM
) -> list[Edge]:
    assert (edge_path_counts is None) == (edges is None)

    # get start nodes
    start_nodes = [seq_graph.seq_nodes[i] for i in range(seq_graph.last_seq_idx if seq_graph.token else 1)]
    assert all([node.node_type == NodeType.RESID_START for node in start_nodes])

    # sample start node
    path = []
    curr_idx, path_count = _sample_edge_idx(start_nodes, edge_path_counts, True, sample_type)
    curr = start_nodes[curr_idx]
    # sample path to include edge (if edge with path)
    if edge_path_counts is not None: 
        edge_found = False 
        last_edge_in_next = False
        while not (edge_found) and (not last_edge_in_next):
            # sample next edge based on edge path counts
            curr_idx, path_count = _sample_edge_idx(
                [edge.tail for edge in curr.out_edges], 
                edge_path_counts, 
                not curr.is_src,
                sample_type
            )
            curr_edge = curr.out_edges[curr_idx]
            # add to path if src to dest edge
            if curr_edge.edge != None:
                path.append(curr_edge.edge)
                # check if edge in edges
                edge_found = curr_edge.edge in edges # sample incedentlly
            # update current node
            curr = curr_edge.tail
            # check if last edge in next node
            if curr.is_src:
                last_edge_in_next = path_count == 1 and _node_edge_in_edges(curr, edges) # must sample last edge
        if last_edge_in_next: # must sample last edge
            curr_edge = next(edge for edge in curr.out_edges if edge.edge in edges)
            curr = curr_edge.tail
            path.append(curr_edge.edge)
    
    # sample rest of path
    while curr.layer != seq_graph.max_layer:
        curr_idx, path_count = _sample_edge_idx([edge.tail for edge in curr.out_edges], sample_type=sample_type)
        cur_edge = curr.out_edges[curr_idx]
        if cur_edge.edge != None:
            path.append(cur_edge.edge)
        curr = cur_edge.tail
    return path

def sample_path_random_walk_rejection( # only used for comparision, should be the same as sample_paths
    seq_graph: SeqGraph,
    complement_edges: set[Edge],
) -> list[Edge]:
    # get start nodes 
    start_nodes = [seq_graph.seq_nodes[i] for i in range(seq_graph.last_seq_idx if seq_graph.token else 1)]
    assert all([node.node_type == NodeType.RESID_START for node in start_nodes])
    start_nodes = [node for node in start_nodes if node.path_count > 0]
    
    # wait until path includes edge in complement_edges
    path = []
    while len(path) == 0 or not any(edge in complement_edges for edge in path):
        path = []
        curr = random.choice(start_nodes) # could be difference here?
        while curr.layer != seq_graph.max_layer:
            valid_edges = [edge for edge in curr.out_edges if edge.tail.path_count > 0]
            curr_edge = random.choice(valid_edges)
            if curr_edge.edge != None:
                path.append(curr_edge.edge)
            curr = curr_edge.tail
    return path
    

def sample_paths(
    seq_graph: SeqGraph, 
    n_paths: int, 
    complement_edges: set[Edge],
    sample_type: SampleType=SampleType.RANDOM_WALK
) -> list[list[Edge]]:
    edge_path_counts = get_edge_path_counts(complement_edges, seq_graph)
    return [
        sample_path(seq_graph, edge_path_counts, complement_edges, sample_type) 
        for _ in tqdm(range(n_paths))
    ]


# def visualize_graph(graph:Graph, sort_by_head: bool=True):
#     # Create a new directed graph
#     G = nx.DiGraph()

#     SeqNode = namedtuple('SeqNode', ['destnode', 'seq_idx'])

#     # Add nodes and edges to the graph
#     for source, targets in graph.items():
#         source_seq = SeqNode(destnode=source.dest, seq_idx=source.seq_idx)
#         G.add_node(str(source.dest), layer=source.dest.layer, seq_idx=source.seq_idx, head_idx=source.dest.head_idx)
#         for target in targets:
#             G.add_edge(str(source), str(target))

#     # Set up the plot
#     plt.figure(figsize=(24, 16))
    
#     # Create a custom layout for the graph
#     pos = {}
#     seq_idx_set = sorted(set(data['seq_idx'] for _, data in G.nodes(data=True)))
#     layer_set = sorted(set(data['layer'] for _, data in G.nodes(data=True)))  # No longer reversed
    
#     # Group nodes by layer and seq_idx
#     grouped_nodes = defaultdict(list)
#     for node, data in G.nodes(data=True):
#         grouped_nodes[(data['layer'], data['seq_idx'])].append((node, data))

#     # Calculate layout
#     column_width = 1.5  # Adjust this value to increase horizontal spacing
#     row_height = 5  # Adjust this value to increase vertical spacing
#     max_nodes_in_group = max(len(nodes) for nodes in grouped_nodes.values())
    
#     for (layer, seq_idx), nodes in grouped_nodes.items():
#         x = seq_idx_set.index(seq_idx) * column_width
#         y = (len(layer_set) - 1 - layer_set.index(layer)) * row_height  # Invert y-axis
        
#         # Sort nodes by head_idx (if available) or by node name
#         if sort_by_head:
#             sorted_nodes = sorted(nodes, key=lambda n: (n[1]['head_idx'] if n[1]['head_idx'] is not None else float('inf'), n[0]))
#         else: # sort by Q, K, V, MLP
#             sorted_nodes = sorted(nodes, key=lambda n: (n[0].split('_')[0].split('.')[-1]))
        
#         # Position nodes in a vertical line within their layer and seq_idx group
#         for i, (node, data) in enumerate(sorted_nodes):
#             node_y = y - i * (row_height / (max_nodes_in_group + 1))  # Distribute nodes evenly within the row
#             pos[node] = (x, node_y)

#     # Draw the nodes
#     node_size = 100  # Adjust as needed
#     nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue')

#     # Draw the edges
#     nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=0.5, arrowsize=10)

#     # Add labels to the nodes
#     labels = {node: f"{node.split('_')[0]}" for node in G.nodes()}
#     nx.draw_networkx_labels(G, pos, labels, font_size=6)

#     # Add path counts as labels on the nodes (uncomment if needed)
#     # path_count_labels = {str(node): f"Paths: {count}" for node, count in path_counts.items()}
#     # nx.draw_networkx_labels(G, pos, path_count_labels, font_size=4, font_color='red')

#     plt.title("Graph Visualization with Corrected Layer Spacing")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()