from enum import Enum
from functools import partial

import torch 
import torch as t
from auto_circuit.data import PromptPairBatch
from auto_circuit.utils.tensor_ops import indices_vals
from auto_circuit.utils.tensor_ops import (
    batch_answer_diffs, 
    batch_answer_max_diffs, 
    batch_answer_vals
)


class GradFunc(Enum):
    LOGIT = "logit"
    PROB = "prob"
    LOGPROB = "logprob"
    LOGIT_EXP = "logit_exp"

class AnswerFunc(Enum):
    AVG_DIFF = "avg_diff"
    MAX_DIFF = "max_diff"
    AVG_VAL = "avg_val"
    MSE = "mse"


GRAD_FUNC_DICT = {
    GradFunc.LOGIT: lambda x: x,
    GradFunc.PROB: partial(t.softmax, dim=-1),
    GradFunc.LOGPROB: partial(t.log_softmax, dim=-1),
    GradFunc.LOGIT_EXP: lambda x: t.exp(x) / t.exp(x).sum(dim=-1, keepdim=True).detach(),
}

ANSWER_FUNC_DICT = {
    AnswerFunc.AVG_DIFF: batch_answer_diffs,
    AnswerFunc.MAX_DIFF: batch_answer_max_diffs,
    AnswerFunc.AVG_VAL: batch_answer_vals,
    AnswerFunc.MSE: lambda vals, batch: t.nn.functional.mse_loss(vals, batch.answers)
}

def get_score_func(grad_func: GradFunc, answer_func: AnswerFunc):
    grad_func = GRAD_FUNC_DICT[grad_func]
    answer_func = ANSWER_FUNC_DICT[answer_func]
    return lambda vals, batch: answer_func(grad_func(vals), batch)