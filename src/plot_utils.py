import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "linear_regression_mamba": [
        "Mamba",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "retrieval": [
        "Transformer",
    ],
    "sparse_parity": [
        "Transformer",
    ],
    "filter_linear_regression": [
        "Transformer",
        "Least Squares",
    ],
    "filter_relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "filter_scale_linear_regression": [
        "Transformer",
    ],
    "filter_ortho_linear_regression": [
        "Transformer",
    ]
}


def basic_plot(metrics, models=None, trivial=1.0, colors=None):
    fig, ax = plt.subplots(1, 1)
    # ax.set_facecolor('#F8F8F8')

    if models is not None:
        metrics = {k: metrics[k] for k in models}
        if colors is not None:
            assert len(models) == len(colors)
            colors = {k: colors[i] for i, k in enumerate(models)}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=(colors[name] if colors else palette[color % 10]), lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, color=(colors[name] if colors else None), alpha=0.2)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = "Transformer"
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                elif "mamba" in model_name:
                    model_name = "Mamba"
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = conf.training.curriculum.points.end
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                elif "decision_tree" in r.task:
                    normalization = 1
                elif "retrieval" in r.task:
                    normalization = 1
                elif "sparse_parity" in r.task:
                    normalization = 1

                for k, v in m.items():
                    if not isinstance(v, float):
                        v = v[:xlim]
                        v = [vv / normalization for vv in v]
                    else:
                        v = v / normalization
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics
