from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm", "rnn", "mamba", "s4-mamba", "s4d", "tf-mamba"])),
    "n_positions": merge(tinteger, default(256)),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, default(8)),
    "interleave": merge(tboolean, default(True)),
    "vocab_size": merge(tinteger, default(-1)),
    "mixed_attn": merge(tstring, nullable, default(None)), # Hybrid options: "standard" or "mambaformer"
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "linear_regression",
    "filter_linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "filter_relu_2nn_regression",
    "filter_scale_linear_regression",
    "filter_ortho_linear_regression",
    "decision_tree",
    "retrieval",
    "sparse_parity",
    "token_induction_head",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "data_sampler_kwargs": merge(tdict, nullable, default(None)),
    "do_parallel": merge(tinteger, nullable, default(None)), 
    "device_batch_size": merge(tinteger, nullable, default(None)), 
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed([
        "gaussian",
        "gaussian_for_retrieval",
        "sparse_parity_sampler",
        "filter",
        "unique_key_tokens_fixed_query",
        "unique_key_tokens_random_query",
    ])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "gradient_clip": merge(tfloat, default(10.0)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
