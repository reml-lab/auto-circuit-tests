from auto_circuit.tasks import Task, TASK_DICT 
from auto_circuit.tasks import (
    IOI_TOKEN_CIRCUIT_TASK, 
    IOI_COMPONENT_CIRCUIT_TASK,
    DOCSTRING_COMPONENT_CIRCUIT_TASK, 
    DOCSTRING_TOKEN_CIRCUIT_TASK
)
from auto_circuit.metrics.official_circuits.circuits.ioi_official import (
    ioi_head_based_official_edges,
    ioi_true_edges

)

from auto_circuit_tests.score_funcs import GradFunc, AnswerFunc

DOCSTRING_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Docstring Autoencoder Component Circuit",
    name="Docstring",
    _model_def="pythia-70m-deduped",
    _dataset_name="docstring_prompts",
    batch_size=2,
    batch_count=128,
    _true_edge_func=None,
    token_circuit=False,
    autoencoder_input="resid_delta_mlp",
    autoencoder_max_latents=200,
    autoencoder_pythia_size="2_32768",
    autoencoder_prune_with_corrupt=False,
)
TASK_DICT.update({
    DOCSTRING_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK.key:
    DOCSTRING_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK
})


IOI_TOKEN_CIRCUIT_TASK: Task = Task(
    key="Indirect Object Identification Token Circuit",
    name="Indirect Object Identification",
    _model_def="gpt2-small",
    _dataset_name="ioi/ioi_vanilla_template_prompts",
    batch_size=64,
    batch_count=4,
    _true_edge_func=ioi_head_based_official_edges,
    token_circuit=True,
)
TASK_DICT.update({
    IOI_TOKEN_CIRCUIT_TASK.key:
    IOI_TOKEN_CIRCUIT_TASK
})

IOI_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Indirect Object Identification Component Circuit",
    name="Indirect Object Identification",
    _model_def="gpt2-small",
    _dataset_name="ioi/ioi_prompts",
    batch_size=64,
    batch_count=4,
    _true_edge_func=ioi_true_edges,
    token_circuit=False,
)
TASK_DICT.update({
    IOI_COMPONENT_CIRCUIT_TASK.key:
    IOI_COMPONENT_CIRCUIT_TASK
})

TASK_TO_OUTPUT_ANSWER_FUNCS = {
    IOI_TOKEN_CIRCUIT_TASK.key: (GradFunc.LOGIT, AnswerFunc.MAX_DIFF), 
    IOI_COMPONENT_CIRCUIT_TASK.key: (GradFunc.LOGIT, AnswerFunc.MAX_DIFF),
    DOCSTRING_COMPONENT_CIRCUIT_TASK.key: (GradFunc.LOGIT, AnswerFunc.MAX_DIFF),
    DOCSTRING_TOKEN_CIRCUIT_TASK.key: (GradFunc.LOGIT, AnswerFunc.MAX_DIFF),
}




