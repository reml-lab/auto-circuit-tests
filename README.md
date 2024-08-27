# On the Properties of Neural Circuits Derived from Optimally Thresholded Edge Attribution Scores

Ciruit hypothesis tests from [Shi et al. 2024](https://openreview.net/pdf?id=ibSNv9cldu), 
implemented for [Auto-Circuit](https://github.com/UFO-101/auto-circuit), and code for 
reproducing experiments in (TODO add paper link after review period)

## Getting Started
Clone the repository (using --recursive to pull the proper branch of the auto-circuit 
submodule)

```
git clone --recursive git@github.com:reml-lab/auto-circuit-tests.git
```

Install project and dependencies 
```
pip install -e .
```

Open [find_and_test_circuit.ipynb](find_and_test_circuit.ipynb), play around with experiment configurations in `Config`, and generally explore the worflow 
```python 
class Config: 
    task: str = "Docstring Component Circuit"
    use_abs: bool = True
    ablation_type: Union[AblationType, str] = AblationType.TOKENWISE_MEAN_CORRUPT
    grad_func: Optional[Union[GradFunc, str]] = GradFunc.LOGIT
    answer_func: Optional[Union[AnswerFunc, str]] = AnswerFunc.MAX_DIFF
    ig_samples: int = 10
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
```

## Reproducing Experiments 
We use submitit to run experiments from a jupyter notebook. Open 
[run_experiments.ipynb](run_experiments.ipynb), 

edit the submitit code with to work 
with your cluster/available resources:
```python
# setup the executor
out_dir = repo_path_to_abs_path(OUTPUT_DIR / "hypo_test_out_logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
out_dir.mkdir(exist_ok=True, parents=True)
executor = submitit.AutoExecutor(folder=out_dir)
num_jobs_parallel = 8
executor.update_parameters(
    timeout_min=60*24,
    mem_gb=40,
    gres="gpu:1",
    cpus_per_task=8,
    nodes=1,
    slurm_qos="high", 
    slurm_array_parallelism=num_jobs_parallel
)
```

run all cells under [Setup Executor and Run](run_experiments.ipynb#"Setup-Executor-and-Run")

After all jobs have completed, run the remaining cells under [Analyze Results](run_experiments.ipynb#"Analyze-Results")