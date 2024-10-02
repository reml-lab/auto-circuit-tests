import torch as t 
from auto_circuit.data import PromptDataLoader
from auto_circuit.types import Edge, BatchOutputs, AblationType, PruneScores   
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.prune_algos.utils import compute_loss 


from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc, get_score_func
from auto_circuit_tests.utils.auto_circuit_utils import run_circuit_with_edge_ablated

def compute_edge_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    prune_scores: PruneScores,
    edges: list[Edge],
    grad_func: GradFunc,
    answer_func: AnswerFunc,
    ablation_type: AblationType,
    model_out: BatchOutputs,
    threshold: float,
) -> dict[Edge, BatchOutputs]:
    score_func = get_score_func(grad_func, answer_func)
    edges_scores: dict[Edge, BatchOutputs] = {}
    for edge in tqdm(edges):
        edge_outs = run_circuit_with_edge_ablated(
            model=model, 
            dataloader=dataloader,
            prune_scores=prune_scores,
            edge=edge,
            ablation_type=ablation_type,
            threshold=threshold,
            to_cpu=False
        )
        # comute scores 
        edge_scores = {
            batch.key: score_func(edge_outs[batch.key], batch, model_out[batch.key]) 
            for batch in dataloader
        }
        edges_scores[edge] = edge_scores
    return edges_scores


def compute_full_model_score(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    model_outs: BatchOutputs,
    grad_func: str,
    answer_func: str,
    device: str
):
    full_model_score = 0
    for batch in dataloader:
        full_model_score -= compute_loss(
            model, 
            batch, 
            grad_func, 
            answer_func, 
            logits=model_outs[batch.key].to(device)
        ).sum().item()
    return full_model_score


def compute_edge_act_prune_scores(
    model: PatchableModel,
    edges_scores: dict[Edge, BatchOutputs],
    full_model_score: float,
) -> PruneScores:
    edge_prune_scores = model.new_prune_scores()
    for mod_name in edge_prune_scores.keys():
        edge_prune_scores[mod_name] += full_model_score
    for edge, edge_scores in edges_scores.items():
            edge_scores = t.cat([v for v in edge_scores.values()])
            edge_prune_scores[edge.dest.module_name][edge.patch_idx] -= edge_scores.mean() # reflects mean computed in compute_loss
    return edge_prune_scores