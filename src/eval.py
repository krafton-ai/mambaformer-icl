import json
import os
import sys
import re
import torch
import yaml


from munch import Munch
from quinine import QuinineArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler
from schema import schema
from filtering_utils import generate_dummy_vectors, zero_pad_ys, dummy_replace


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        model_path = os.path.join(run_path, "state.pt")
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
    
    state_dict = torch.load(model_path)
    if "model_state_dict" in state_dict:
        state_dict = compatible_removal(state_dict["model_state_dict"])
    else:
        state_dict = compatible_removal(state_dict)

    if conf.model.family == "s4":
        state_dict = {key.replace("S4D.", "S4DMamba."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    
    return model, conf


def compatible_removal(model_state_dict):
    # Remove "_backbone.h.0.attn.bias", "_backbone.h.0.attn.masked_bias" from the state_dict
    for layer in range(48):
        model_state_dict.pop(f'_backbone.h.{layer}.attn.bias', None)
        model_state_dict.pop(f'_backbone.h.{layer}.attn.masked_bias', None)
    return model_state_dict


# Functions for evaluation
def eval_batch(conf, model, task_sampler, xs, xs_p=None, inds=None):
    task = task_sampler()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if xs_p is None:
        ys = task.evaluate(xs)
        if "Filter" in task.__class__.__name__ and inds is not None:
            filtering_tasks = ["filter_linear_regression", "filter_relu_2nn_regression", ]
            if conf.training.task == "filter_ortho_linear_regression":
                # zero padding -> replace with au+bv
                ys = zero_pad_ys(model.tokenized, xs, ys)
                xs = dummy_replace(inds, task_sampler(), conf.training.task, xs)
                ys = dummy_replace(inds, task_sampler(), conf.training.task, ys)
            else:
                if conf.training.task == "filter_scale_linear_regression":
                    # replace xs[inds] = xs[inds]*scale
                    xs = dummy_replace(inds, task_sampler(), conf.training.task, xs)

                elif conf.training.task in filtering_tasks:
                    xs = dummy_replace(inds, task_sampler(), conf.training.task, xs)
                    ys = dummy_replace(inds, task_sampler(), conf.training.task, ys)
        if len(ys.shape) == 2:
            ys = zero_pad_ys(model.tokenized, xs, ys)

        pred = model(xs.to(device), ys.to(device)).detach()
        if len(pred.shape) == 2 and len(ys.shape) == 3:
            ys = ys[:,:,0]
        # Transform prediction
        if task.__class__.__name__ == "Retrieval":
            metrics = task.classify(xs.cpu(), ys.cpu(), pred.cpu())
        elif task.__class__.__name__ == "InductionHead": 
            pred_tokens = torch.argmax(pred.cpu(), dim=1)
            metrics = task.classify(xs.cpu(), ys.cpu(), pred_tokens)
        elif "Filter" in task.__class__.__name__ and inds is not None:
            if len(inds.shape) == 1:
                metrics = task.get_metric()(pred[:, ~inds].cpu(), ys[:, ~inds].cpu())
            else:
                metrics = task.get_metric()(pred[~inds].cpu(), ys[~inds].cpu())
        else:
            metrics = task.get_metric()(pred.cpu(), ys.cpu())
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :].to(device), xs_p[:, i:, :].to(device)), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys.cpu())[:, i]

    return metrics


# Functions for generating different kinds of train/test data
def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_noisy_query(data_sampler, n_points, b_size, sigma):
    xs = data_sampler.sample_xs(n_points, b_size, noise_sigma=sigma)

    return xs, None


def gen_by_position(data_sampler, n_points, b_size, query_pos):
    xs = data_sampler.sample_xs(n_points, b_size, query_pos=2*query_pos)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]], device="cuda").sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]], device="cuda").sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape, device="cuda")
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i, device="cuda") for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    if len(metrics[bootstrap_indices].shape) > 2:
        bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
        results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
        results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    conf,
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    if num_eval_examples % batch_size != 0:
        num_eval_examples = (num_eval_examples // batch_size + 1) * batch_size
        print(f"Adapting eval batch size to {num_eval_examples}...")
    
    if "token" in task_name:
        data_sampler = get_data_sampler(data_name, n_dims=n_dims, vocab_size=model.vocab_size, **data_sampler_kwargs)
    else:
        data_sampler = get_data_sampler(data_name, n_dims=n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []

    if prompting_strategy == "fixed_dummy" or prompting_strategy == "wo_dummy":
        generating_func = globals()["gen_standard"]    
    else:
        generating_func = globals()[f"gen_{prompting_strategy}"]

    if prompting_strategy == "by_position":
        for pos in range(n_points):
            metrics_by_position = []
            for i in range(num_eval_examples // batch_size):
                xs, xs_p = generating_func(data_sampler, n_points, batch_size, pos)
                metrics_batch = eval_batch(conf, model, task_sampler, xs, xs_p)
                metrics_by_position.append(metrics_batch)
            all_metrics.append(torch.cat(metrics_by_position, dim=0))
        metrics = torch.cat(all_metrics, dim=1)
    elif prompting_strategy == "noisy_query":
        for sigma in [0.1, 0.4, 0.7, 1.0, 2.0, 3.0, 5.0]:
            metrics_by_sigma = []
            for i in range(num_eval_examples // batch_size):
                xs, xs_p = generating_func(data_sampler, n_points, batch_size, sigma)
                metrics_batch = eval_batch(conf, model, task_sampler, xs, xs_p)
                metrics_by_sigma.append(metrics_batch)
            all_metrics.append(torch.cat(metrics_by_sigma, dim=0))
        metrics = torch.cat(all_metrics, dim=1)
    elif prompting_strategy == "fixed_dummy":
        # fixed dummy location; xs filtering done in eval_batch
        if data_sampler.prob == 0.0:
            inds = None
        else:
            spacing = int(1 // min(data_sampler.prob, 1 - data_sampler.prob))
            inds = torch.zeros(n_points, dtype=torch.bool)
            inds[range(spacing-1, len(inds), spacing)] = True
            if data_sampler.prob > 0.5:
                inds = torch.logical_not(inds)

        for i in range(num_eval_examples // batch_size):
            xs, xs_p = generating_func(data_sampler, n_points, batch_size)
            if data_sampler.prob == 0.0:
                inds = None
            else:
                spacing = int(1 // min(data_sampler.prob, 1 - data_sampler.prob))
                num_dummy_vectors = int(xs.shape[1]*data_sampler.prob)
                gap = xs.shape[1] // num_dummy_vectors

                # need to update indices in data_sampler
                inds = torch.zeros(xs.shape[1], dtype=torch.bool, device=xs.device)
                while spacing < xs.shape[1]:
                    inds[spacing] = True
                    spacing += gap
                
                if data_sampler.prob > 0.5:
                    inds = torch.logical_not(inds)

                data_sampler.inds = inds    # not needed?
                xs = generate_dummy_vectors(task_sampler(), task_name, xs, inds.unsqueeze(0).expand(xs.shape[0], xs.shape[1]))
            
            metrics_batch = eval_batch(conf, model, task_sampler, xs, xs_p, inds=inds)
            all_metrics.append(metrics_batch)
        metrics = torch.cat(all_metrics, dim=0)
    else:
        for i in range(num_eval_examples // batch_size):
            xs, xs_p = generating_func(data_sampler, n_points, batch_size)
            if prompting_strategy == "wo_dummy":
                data_sampler.inds = None
            metrics_batch = eval_batch(conf, model, task_sampler, xs, xs_p, inds=data_sampler.inds)

            all_metrics.append(metrics_batch)
        metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}
    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    if task_name != "linear_regression":
        if task_name in ["retrieval"]:
            evaluation_kwargs["standard"] = {"num_eval_examples": 12800}
            evaluation_kwargs["retrieval"] = {"prompting_strategy": "by_position"}
            evaluation_kwargs["noisy_query"] = {
                "num_eval_examples": 2560,
                "prompting_strategy": "noisy_query"
            }
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        if task_name in ["sparse_parity", "token_induction_head"]:
            evaluation_kwargs["standard"]["task_sampler_kwargs"] = dict(conf.training.task_kwargs)
        if "filter" in task_name:
            evaluation_kwargs["fixed_dummy"] = {
                "prompting_strategy": "fixed_dummy", 
                "data_sampler_kwargs": {"prob": conf.training.data_sampler_kwargs.prob}
            }
            evaluation_kwargs["wo_dummy"] = {
                "prompting_strategy": "wo_dummy", 
            }
            evaluation_kwargs["standard"]["data_sampler_kwargs"] = {"prob": conf.training.data_sampler_kwargs.prob}
            
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_evals(conf, all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}
    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue
            metrics[model.name] = eval_model(conf, model, **kwargs)
        all_metrics[eval_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False, recompute_metrics=False,
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        model = model.cuda().eval()
        all_models = [model]
        if not skip_baselines:
            all_models += models.get_relevant_baselines(conf.training.task)
    evaluation_kwargs = build_evals(conf)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(conf, all_models, evaluation_kwargs, save_path, recompute or recompute_metrics)
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return f"Transformer"
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    # assert len(df) == len(df.run_name.unique())
    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for i, run_id in tqdm(enumerate(os.listdir(run_dir))):
        run_path = os.path.join(run_dir, run_id)
        print(f"Evaluating task {run_path}")

        # Iterate over all files in the directory
        pattern = re.compile(r'model_(\d+).pt')
        model_indices = []
        for filename in os.listdir(run_path):
            match = pattern.match(filename)
            if match:
                index = int(match.group(1))
                model_indices.append(index)
        for step in model_indices:
            if step % 100000 == 0:
                print(f"Evaluating model_index: step {step}")
                metrics = get_run_metrics(run_path, step=step, skip_baselines=True, recompute_metrics=True)