import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks, CausalLinearAttention
from mamba_ssm.models.mixer_seq_simple import MixerModel


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            interleave=getattr(conf, 'interleave', True),
            vocab_size=getattr(conf, 'vocab_size', -1),
        )
    elif conf.family == "rnn":
        model = RNNModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            interleave=getattr(conf, 'interleave', True),
            vocab_size=getattr(conf, 'vocab_size', -1),
        )
    elif conf.family == "tf-mamba":
        model = MambaMixerModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            interleave=getattr(conf, 'interleave', True),
            vocab_size=getattr(conf, 'vocab_size', -1),
            mixed_attn=getattr(conf, 'mixed_attn', None),
            n_positions=conf.n_positions,
        )
    elif conf.family == "mamba":
        model = MambaMixerModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            interleave=getattr(conf, 'interleave', True),
            vocab_size=getattr(conf, 'vocab_size', -1),
        )
    elif conf.family == "s4-mamba":
        model = MambaMixerModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            interleave=getattr(conf, 'interleave', True),
            vocab_size=getattr(conf, 'vocab_size', -1),
            s4=True,
        )
    elif conf.family == "s4d":
        model = MambaMixerModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            interleave=getattr(conf, 'interleave', True),
            vocab_size=getattr(conf, 'vocab_size', -1),
            s4=True,
            mamba_style_block=False,
        )
    elif conf.family == "mlp-mixer":
        model = MLPMixerModel(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            interleave=getattr(conf, 'interleave', True),
            vocab_size=getattr(conf, 'vocab_size', -1),
            max_seq_len=conf.n_positions,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
        "retrieval": [],
        "sparse_parity": [],
        "token_induction_head": [],
        "filter_linear_regression": [
            (LeastSquaresModel, {}),
        ],
        "filter_relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "filter_scale_linear_regression": [
            (LeastSquaresModel, {}),
        ],
        "filter_ortho_linear_regression": [
            (LeastSquaresModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class BaseModel(nn.Module):
    def __init__(
        self,
        n_dims=20,
        n_embd=128,
        n_layer=12,
        interleave=True,
        vocab_size=-1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.interleave = interleave
        self.tokenized = (vocab_size > 0)
        assert not (self.interleave and self.tokenized)

        self.n_dims = -1 if self.tokenized else n_dims
        self.vocab_size = vocab_size if vocab_size > 0 else 50257
        print(f"Interleaving samples: {interleave}. Tokenized: {self.tokenized}. Vocab size: {vocab_size}.")

        self._read_in = nn.Embedding(vocab_size, n_embd) if self.tokenized else nn.Linear(n_dims, n_embd)
        self._backbone = None
        if self.tokenized:
            self._read_out = nn.Linear(n_embd, vocab_size, bias=False)
            self.tie_weights()
        elif self.interleave:
            self._read_out = nn.Linear(n_embd, 1)
        else:
            self._read_out = nn.Linear(n_embd, n_dims)

    def _combine(self, xs_b, ys_b):
        """
        Interleaves the x's and the y's into a single sequence.

        Returns:
        zs: Input to _read_in linear layer. Shape depends on tokenization.
            (bsize, 2*points) if tokenized into input_ids
            (bsize, 2 * points, dim) if points are R^dim vectors
        """
        if self.tokenized:
            (bsize, points), dim = xs_b.shape, 1
        else:
            bsize, points, dim = xs_b.shape
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points) if self.tokenized else zs.view(bsize, 2 * points, dim)
        return zs

    def tie_weights(self):
        """Ties projections in and out to be the same."""
        self._read_out.weight = self._read_in.weight

    def compile(self, xs, ys, inds):
        """Determines which points to predict for. Then combines xs, ys to zs."""
        if inds is None and self.interleave:
            inds = torch.arange(ys.shape[1])
        elif not self.interleave:  # Only predict on last vector
            inds = -1
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine(xs, ys) if self.interleave else xs
        return zs, inds

    def masked_predictions(self, prediction, inds):
        """
        Filters the final prediction into the predictions that matter.

        Parameters:
        prediction: Final predictions for all points.
        inds: Indices of which predictions matter for the objective; e.g., for interleave, this is all even indices.
        """
        # Filter by inds
        if self.interleave:
            return prediction[:, ::2, 0][:, inds]  # predict only on xs
        return prediction[:, inds, :]
    
    def forward(self):
        raise NotImplementedError("The forward method must be implemented by the subclass")


class TransformerModel(BaseModel):
    def __init__(
        self, 
        n_dims=20, 
        n_positions=101, 
        n_embd=128, 
        n_layer=12, 
        n_head=4, 
        interleave=True,
        vocab_size=-1,
    ):
        super().__init__(
            n_dims=n_dims,
            n_embd=n_embd,
            n_layer=n_layer,
            interleave=interleave,
            vocab_size=vocab_size,
        )

        self.n_positions = n_positions
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            vocab_size=self.vocab_size,
        )

        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self._backbone = GPT2Model(configuration)


    def forward(self, xs, ys, inds=None):
        zs, inds = self.compile(xs, ys, inds)
        
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        prediction = self.masked_predictions(prediction, inds)
        return prediction



class MambaMixerModel(BaseModel):
    def __init__(
        self,
        n_dims=20,
        n_embd=128,
        n_layer=12,
        interleave=True,
        vocab_size=-1,
        s4=False,
        mamba_style_block=True,
        mixed_attn=None,
        n_positions=-1,
    ):
        super().__init__(
            n_dims=n_dims,
            n_embd=n_embd,
            n_layer=n_layer,
            interleave=interleave,
            vocab_size=vocab_size,
        )
        self.mixed_attn = mixed_attn
        if mixed_attn == "standard":
            assert n_positions > 0
            self.wpe = nn.Embedding(n_positions, n_embd)

        self.name = f"{'s4' if s4 else 'mamba'}_embd={n_embd}_layer={n_layer}" 
        self._backbone = MixerModel(
            d_model=n_embd,
            n_layer=n_layer,
            s4=s4,
            mamba_style_block=mamba_style_block,
            mixed_attn=mixed_attn,
            block_size=n_positions,
            vocab_size=1, # unused
        )


    def forward(self, xs, ys, inds=None):
        zs, inds = self.compile(xs, ys, inds)
        embeds = self._read_in(zs)
        if self.mixed_attn == "standard":
            pos = torch.arange(0, zs.shape[1], dtype=torch.long, device=zs.device)
            pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)
            embeds += pos_emb
        output = self._backbone(input_ids=None, inputs_embeds=embeds)
        prediction = self._read_out(output)

        return self.masked_predictions(prediction, inds)


class RNNModel(BaseModel):
    def __init__(
        self,
        n_dims=20,
        n_embd=128,
        n_layer=12,
        interleave=True,
        vocab_size=-1,
    ):
        super().__init__(
            n_dims=n_dims,
            n_embd=n_embd,
            n_layer=n_layer,
            interleave=interleave,
            vocab_size=vocab_size,
        )

        self.name = f"RNN_embd={n_embd}_layer={n_layer}"
        # Creating RNN layers
        self.rnn_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.rnn_layers.append(nn.RNN(
                input_size=n_embd,
                hidden_size=n_embd,
                batch_first=True,
                nonlinearity='relu',
            ))
            self.ln_layers.append(nn.LayerNorm(n_embd))

    def forward(self, xs, ys, inds=None):
        zs, inds = self.compile(xs, ys, inds)

        output = self._read_in(zs)
        for i in range(self.n_layer):
            output, _ = self.rnn_layers[i](output)
            output = self.ln_layers[i](output)
        prediction = self._read_out(output)

        return self.masked_predictions(prediction, inds)


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


class MLPMixerModel(BaseModel):
    def __init__(
        self,
        n_dims=20,
        n_positions=101,
        n_embd=128,
        n_layer=12,
        n_head=8,
        interleave=True,
        vocab_size=-1,
        max_seq_len=41,     # conf.training.curriculum.points.end
    ):
        super().__init__(
            n_dims=n_dims,
            n_embd=n_embd,
            n_layer=n_layer,
            interleave=interleave,
            vocab_size=vocab_size,
        )

        self.name = f"mlp_mixer_embd={n_embd}_layer={n_layer}"
        self._backbone = nn.ModuleList([
            CausalLinearAttention(n_embd, n_head, max_seq_len) 
        for i in range(n_layer)])
        
        self.wpe = nn.Embedding(max_seq_len, n_embd)
        
    
    def forward(self, xs, ys, inds=None):
        zs, inds = self.compile(xs, ys, inds)
        position_ids = torch.arange(zs.shape[1], dtype=torch.long, device=zs.device)
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        
        embeds = self._read_in(zs) + position_embeds

        for i in range(self.n_layer):
            embeds = self._backbone[i](xs=embeds)

        predictions = self._read_out(embeds)
        return self.masked_predictions(predictions, inds)
    

# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)


if __name__ == "__main__":
    model = MLPMixerModel()

    xs = torch.randn(64, 11, 20)
    ys = torch.randn(64, 11)

    print(model(xs, ys).shape)