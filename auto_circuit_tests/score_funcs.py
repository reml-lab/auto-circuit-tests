from enum import Enum
from functools import partial
from typing import Callable, Optional

import torch 
import torch as t
from auto_circuit.data import PromptPairBatch
from auto_circuit.utils.tensor_ops import indices_vals
from auto_circuit.utils.tensor_ops import (
    batch_answer_diffs, 
    batch_answer_max_diffs, 
    batch_answer_vals,
    batch_kl_divs,
    batch_js_divs
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
    KL_DIV = "kl_div"
    JS_DIV = "js_div"


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
    AnswerFunc.MSE: lambda vals, batch: t.nn.functional.mse_loss(vals, batch.answers),
    AnswerFunc.KL_DIV: lambda input, target: -batch_kl_divs(input, target), # negative because we want to maximize
    AnswerFunc.JS_DIV: lambda input, target: -batch_js_divs(input, target)
}

DIV_ANSWER_FUNCS = {AnswerFunc.KL_DIV, AnswerFunc.JS_DIV}

def compute_scores(
    model_out: t.Tensor,
    batch: Optional[PromptPairBatch]=None,
    full_model_out: Optional[t.Tensor]=None,
    grad_func: GradFunc = GradFunc.LOGIT,
    answer_func: AnswerFunc = AnswerFunc.AVG_VAL
) -> t.Tensor:
    vals = GRAD_FUNC_DICT[grad_func](model_out)
    if answer_func in DIV_ANSWER_FUNCS:
        ans_func_args = GRAD_FUNC_DICT[GradFunc.LOGPROB](full_model_out)
    else: 
        ans_func_args = batch
    return ANSWER_FUNC_DICT[answer_func](vals, ans_func_args)

def get_score_func(
        grad_func: GradFunc, answer_func: AnswerFunc
) -> Callable[[t.Tensor, Optional[PromptPairBatch], Optional[t.Tensor]], t.Tensor]:
    return partial(compute_scores, grad_func=grad_func, answer_func=answer_func)

