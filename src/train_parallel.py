import os
import json
from random import randint
import uuid
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter
from quinine import QuinineArgumentParser
from time import strftime, localtime
from tqdm import tqdm
import numpy as np

from eval import get_run_metrics, eval_model
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from util import Logger
from filtering_utils import zero_pad_ys, dummy_replace

from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
import torch.multiprocessing as mp


torch.backends.cudnn.benchmark = True


def train_step(rank, iter, model, xs, ys, optimizer, loss_func, max_grad_norm=float('inf'), filtering_inds=None):
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
        output[filtering_inds] = torch.ones_like(output[filtering_inds]) # not used; only for logging
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


##############################################################################################################################################


def train_parallel(rank, world_size, model, args):

    global_rank = rank
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:33447",
        rank=global_rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # logging setup
    if dist.get_rank() == 0:
        writer = SummaryWriter(os.path.join(args.out_dir, "tensorboard"))
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


    model = FSDP(model).cuda(rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)


    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0

    ######################################################
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        with FSDP.summon_full_params(model):
            state = torch.load(state_path)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            starting_step = state["train_step"]
            for i in range(state["train_step"] + 1):
                curriculum.update()
    ######################################################

    n_dims = model.n_dims
    # bsize = args.training.batch_size
    bsize = args.training.device_batch_size

    if "token" in args.training.task:
        data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, vocab_size=model.vocab_size)
    elif args.training.data_sampler_kwargs:
        data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **args.training.data_sampler_kwargs)
    else:
        data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)

    #######################################################
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    #######################################################

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


        ############################################################
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        ############################################################
        

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
            rank,
            i,
            model, 
            xs.cuda(rank),
            ys.cuda(rank),
            optimizer,
            loss_func,
            args.training.gradient_clip,
            filtering_inds=filtering_inds
        )

        if len(output.shape) == 2 and len(ys.shape) == 3:
            ys_loss = ys[:,:,0]
        else:
            ys_loss = ys


        ####################################################################################

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        # Calculate pointswise loss including dummy ones: output=ys=1 for dummies
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys_loss.cuda(rank)).mean(dim=0)

        if model.interleave and len(point_wise_loss.shape) >= 1:
            # point_wise_loss
            dist.all_reduce(point_wise_loss, op=dist.ReduceOp.SUM)
            avg_point_wise_loss = point_wise_loss / world_size

            point_wise_dict = dict(zip(range(avg_point_wise_loss.shape[0]), avg_point_wise_loss.cpu().numpy()))
            point_wise_dict = {k: float(v) if isinstance(v, np.float32) else v for k, v in point_wise_dict.items()}

        ########################################################################################################################
        
        ##################################################################
        temporary_grad_norm = torch.tensor(grad_norm, device=rank) # 추후 변경 필요, 속도를 위해서
        dist.all_reduce(temporary_grad_norm, op=dist.ReduceOp.SUM)
        avg_grad_norm = temporary_grad_norm.item() / world_size

        excess_loss = torch.tensor( loss / baseline_loss, device=rank) # 추후 변경 필요, 속도를 위해서
        dist.all_reduce(excess_loss, op=dist.ReduceOp.SUM)
        avg_excess_loss = excess_loss.item() / world_size

        temporary_loss = torch.tensor( loss, device=rank )
        dist.all_reduce(temporary_loss, op=dist.ReduceOp.SUM)
        avg_loss = temporary_loss.item() / world_size
        ##################################################################

        if (dist.get_rank() == 0) and ( grad_norm > args.training.gradient_clip ):
            writer.add_scalar("gradient/norm", avg_grad_norm, i)


        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            if dist.get_rank() == 0:
                writer.add_scalar("overall_loss/train", avg_loss, i)
                writer.add_scalar("excess_loss/train", avg_excess_loss, i)
                writer.add_scalar("gradient/norm", avg_grad_norm, i)

                if model.interleave and len(point_wise_loss.shape) >= 1:
                    for k in point_wise_dict:
                        # Not comparable against different filtering probabilities
                        writer.add_scalar(f"pointwise/loss_@{k}_shot", point_wise_dict[k], i)

        ########################################################################################################################


        curriculum.update()

        pbar.set_description(f"loss {loss}")
        
        if i % args.training.save_every_steps == 0 and not args.test_run:
    
            dist.barrier()

            # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):

            #     cpu_state = model.state_dict()
            #     if dist.get_rank() == 0:  # Only rank 0 performs the save
            #         training_state = {
            #             "model_state_dict": cpu_state,
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "train_step": i,
            #         }
            #         torch.save(training_state, state_path)

            if (
                args.training.keep_every_steps > 0
                and i % args.training.keep_every_steps == 0
                and not args.test_run
                and i > 0
            ):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    cpu_state = model.state_dict()
                    if dist.get_rank() == 0:  # Only rank 0 performs the save
                        # Save the state dict and optimizer state
                        torch.save({
                                "model_state_dict": cpu_state,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_step": i,
                            },
                            os.path.join(args.out_dir, f"model_{i}.pt")
                        )


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
            
            eval_keys = eval_metric.keys() 
            avg_eval_metric = {}

            for k in eval_keys:
                if isinstance(eval_metric[k], list):
                    for value in eval_metric[k]:
                        measured_value = torch.tensor(value, device=rank)
                        dist.all_reduce(measured_value, op=dist.ReduceOp.SUM)
                        avg_measured_value = measured_value.item() / world_size
                        avg_eval_metric[k] = avg_eval_metric.get(k, []) + [avg_measured_value]
                else:
                    measured_value = torch.tensor( eval_metric[k], device=rank)
                    dist.all_reduce(measured_value, op=dist.ReduceOp.SUM)
                    avg_measured_value = measured_value.item() / world_size
                    avg_eval_metric[k] = avg_measured_value

            if dist.get_rank() == 0:
                if isinstance(eval_metric['mean'], list):        
                    writer.add_scalar("error_rate/eval", avg_eval_metric['mean'][-1], i)
                elif isinstance(eval_metric['mean'], float):
                    writer.add_scalar("error_rate/eval", avg_eval_metric['mean'], i)
                else:
                    raise ValueError("eval metric threw unexpected datatype")

        if dist.get_rank() == 0:
            writer.flush()
            writer.close()

##############################################################################################################################################


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100

    model = build_model(args.model)
    model.train()

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        world_size = num_gpus
        args.training.device_batch_size = int( args.training.batch_size / num_gpus )
        mp.spawn(train_parallel, nprocs=num_gpus, args=(world_size, model, args))
    else:
        print("CUDA is not available. No GPUs detected.")

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


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