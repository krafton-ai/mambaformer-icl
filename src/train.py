import os
import json
import uuid
import torch
import yaml
# import wandb

from torch.utils.tensorboard import SummaryWriter
from quinine import QuinineArgumentParser
from tqdm import tqdm
import numpy as np
from random import randint
from time import strftime, localtime

from eval import get_run_metrics, eval_model
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from util import Logger
from filtering_utils import zero_pad_ys, dummy_replace


torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, max_grad_norm=float('inf'), filtering_inds=None):
    optimizer.zero_grad()
    output = model(xs, ys)
    # since we zero-padded before, this step is needed
    # to ensure the shape of pred and ys match
    if len(output.shape) == 2 and len(ys.shape) == 3:
        ys_loss = ys[:,:,0]
    else:
        ys_loss = ys

    if filtering_inds is not None:
        loss = loss_func(output[~filtering_inds], ys_loss[~filtering_inds])
    else:
        loss = loss_func(output, ys_loss)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.detach().item(), output.detach(), grad_norm.cpu().item()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    # logging setup
    writer = SummaryWriter(os.path.join(args.out_dir, "wandb"))
    writer.add_hparams({
        "lr": args.training.learning_rate,
        "gradient_clip": args.training.gradient_clip,
        "train_steps": args.training.train_steps,
        "bsize": args.training.batch_size,
        "n_dims": args.model.n_dims,
        "n_points": args.training.curriculum.points.end,
        "n_embd": args.model.n_embd,
        "n_layer": args.model.n_layer,
        "n_head": getattr(args.model, 'n_head', -1),
    }, {})

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    if "token" in args.training.task:
        data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, vocab_size=model.vocab_size)
    elif args.training.data_sampler_kwargs:
        data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **args.training.data_sampler_kwargs)
    else:
        data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}
        filtering_inds = None

        if "sparse_linear" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        
        # this needs to be placed after evaluate()
        # indices are populated in each evaluate() call
        if "filter" in args.training.task:
            filtering_inds = data_sampler.inds
            
        filtering_tasks = ["filter_linear_regression", "filter_relu_2nn_regression", ]
        if args.training.task == "filter_ortho_linear_regression":
            # zero padding -> replace with au+bv
            ys = zero_pad_ys(model.tokenized, xs, ys)
            xs = dummy_replace(data_sampler.inds, task_sampler(), args.training.task, xs)
            ys = dummy_replace(data_sampler.inds, task_sampler(), args.training.task, ys)
        else:
            if args.training.task == "filter_scale_linear_regression":
                # replace xs[inds] = xs[inds]*scale
                xs = dummy_replace(data_sampler.inds, task_sampler(), args.training.task, xs)

            elif args.training.task in filtering_tasks:
                xs = dummy_replace(data_sampler.inds, task_sampler(), args.training.task, xs)
                ys = dummy_replace(data_sampler.inds, task_sampler(), args.training.task, ys)
            if len(ys.shape) == 2:
                ys = zero_pad_ys(model.tokenized, xs, ys)

        loss_func = task.get_training_metric()
        loss, output, grad_norm = train_step(
            model, 
            xs.cuda(),
            ys.cuda(),
            optimizer,
            loss_func,
            max_grad_norm=args.training.gradient_clip,
            filtering_inds=filtering_inds
        )

        if len(output.shape) == 2 and len(ys.shape) == 3:
            ys_loss = ys[:,:,0]
        else:
            ys_loss = ys

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )


        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            writer.add_scalar("overall_loss/train", loss, i)
            writer.add_scalar("excess_loss/train", loss / baseline_loss, i)
            writer.add_scalar("gradient/norm", grad_norm, i)
            writer.add_scalar("gradient/lr", optimizer.param_groups[0]['lr'], i)

            # Calculate pointswise loss including dummy ones: output=ys=1 for dummies
            point_wise_loss_func = task.get_metric()
            point_wise_loss = point_wise_loss_func(output, ys_loss.cuda()).mean(dim=0)

            if model.interleave and len(point_wise_loss.shape) >= 1:
                point_wise_dict = dict(zip(range(point_wise_loss.shape[0]), point_wise_loss.cpu().numpy()))
                point_wise_dict = {k: float(v) if isinstance(v, np.float32) else v for k, v in point_wise_dict.items()}
                for k in point_wise_dict:
                    # Not comparable against different filtering probabilities
                    writer.add_scalar(f"pointwise/loss_@{k}_shot", point_wise_dict[k], i)
        curriculum.update()

        pbar.set_description(f"{args.wandb.name}; loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

            # eval (measure no-dummy for filtering tasks)
            eval_metric = eval_model(
                args,
                model,
                args.training.task,
                args.training.data,
                args.model.n_dims,
                curriculum.n_points,
                prompting_strategy="standard",
                num_eval_examples=1280,
                batch_size=64,
                data_sampler_kwargs={},
                task_sampler_kwargs=args.training.task_kwargs,
            )
            
            if isinstance(eval_metric['mean'], list):        
                writer.add_scalar("error_rate/eval", eval_metric['mean'][0], i)
            elif isinstance(eval_metric['mean'], float):
                writer.add_scalar("error_rate/eval", eval_metric['mean'], i)
            else:
                raise ValueError("eval metric threw unexpected datatype")
        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_step": i,
                },
                os.path.join(args.out_dir, f"model_{i}.pt")
            )
        
        writer.flush()
        writer.close()


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    # else:
    #     # logging
    #     wandb.init(
    #         dir=args.out_dir,
    #         project=args.wandb.project,
    #         entity=args.wandb.entity,
    #         config=args.__dict__,
    #         notes=args.wandb.notes,
    #         name=args.wandb.name,
    #         resume=True,
    #     )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir, skip_baselines=True)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = args.wandb.name + "_" + strftime("%Y-%m-%d-%H:%M:%S", localtime())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
        print(f"Saving directory: {out_dir}")

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)