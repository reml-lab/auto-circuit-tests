import torch as t 
from auto_circuit.data import PromptDataLoader
from auto_circuit.types import Edge, BatchOutputs, AblationType, PruneScores   
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.custom_tqdm import tqdm


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