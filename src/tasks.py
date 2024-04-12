import math
import torch


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()
ce_loss = torch.nn.CrossEntropyLoss()

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def squared_norm_error(ys_pred, ys, dim=-1):
    return (ys - ys_pred).square().sum(dim=dim).unsqueeze(dim=dim)

def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()

def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "retrieval": Retrieval,
        "sparse_parity": SparseParity,
        "token_induction_head": InductionHead,
        "filter_linear_regression": FilterLinearRegression,
        "filter_relu_2nn_regression": FilterRelu2nnRegression,
        "filter_scale_linear_regression": FilterScaleLinearRegression,
        "filter_ortho_linear_regression": FilterOrthoLinearRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        if "token_induction_head" in task_name:
            return lambda **args: task_cls(
                batch_size,
                pool_dict,
                **args,
                **kwargs
            )
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class Retrieval(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        super(Retrieval, self).__init__(n_dims, batch_size, pool_dict, seeds)

    def evaluate(self, xs_b):
        # xs_b: (self.b_size, n_points, self.n_dims)
        assert xs_b.shape[0] == self.b_size and xs_b.shape[2] == self.n_dims
        assert xs_b.shape[1] % 2 == 1

        keys_b, values_b = xs_b[:, :-1:2, :], xs_b[:, 1::2, :]
        query_b = xs_b[:, -1, :].unsqueeze(dim=1)
        inner_products = torch.bmm(query_b, keys_b.transpose(1, 2)).squeeze(1)
        _, retrieval_inds = torch.max(inner_products, dim=1)
        retrieval_inds = retrieval_inds.view(-1, 1, 1).expand(-1, -1, values_b.size(-1))
        ys_b = torch.gather(values_b, 1, retrieval_inds).squeeze(1)

        # ys_b: (b_size, n_dims)
        return ys_b
    
    def classify(self, xs_b, ys_b, pred):
        keys_b, values_b = xs_b[:, :-1:2, :], xs_b[:, 1::2, :]
        query_b, pred_b = ys_b.unsqueeze(dim=1), pred.unsqueeze(dim=1)

        inner_products = torch.bmm(query_b, keys_b.transpose(1, 2)).squeeze(1)
        _, gold_inds = torch.max(inner_products, dim=1)
        inner_products = torch.bmm(pred_b, keys_b.transpose(1, 2)).squeeze(1)
        _, pred_inds = torch.max(inner_products, dim=1)
        match_error = (1-(gold_inds == pred_inds).float()).unsqueeze(1)
        return match_error

    @staticmethod
    def get_metric():
        return squared_norm_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class InductionHead(Task):
    def __init__(self, batch_size, vocab_size=128, chain_depth=1, pool_dict=None, seeds=None):
        super(InductionHead, self).__init__(1, batch_size, pool_dict, seeds)
        self.vocab_size = vocab_size
        self.chain_depth = chain_depth

    def evaluate(self, xs_b):
        # xs_b: (self.b_size, n_points). The last point should always be 0 (retrieval token)
        assert xs_b.shape[0] == self.b_size and len(xs_b.shape) == 2
        assert xs_b.shape[1] % 2 == 1

        ys_b = xs_b[:, -1]
        for i in range(self.chain_depth):
            # only look at keys (not values)
            mask = xs_b[:, ::2] == ys_b[:, None]
            # Finding the first index of zero in each 1D tensor
            retrieved_inds = torch.argmax(mask.int(), dim=1) * 2 + 1
            ys_b = xs_b[torch.arange(self.b_size), retrieved_inds]

        return ys_b
    
    def classify(self, xs_b, ys_b, pred):
        match_error = (1-(ys_b == pred).float()).unsqueeze(1)
        return match_error

    @staticmethod
    def get_metric():
        return ce_loss

    @staticmethod
    def get_training_metric():
        return ce_loss
    

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class FilterLinearRegression(LinearRegression):
    pass

class FilterScaleLinearRegression(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, dummy_scale=0.01):
        super(FilterScaleLinearRegression, self).__init__(n_dims, batch_size, pool_dict=None, seeds=None, scale=1)
        self.dummy_scale = dummy_scale

class FilterOrthoLinearRegression(LinearRegression):
    pass

class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class FilterRelu2nnRegression(Relu2nnRegression):
    pass


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class SparseParity(Task):
    # k = num_indices
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, k=None):
        super(SparseParity, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert k is not None
        self.k = k
        self.indices = torch.randperm(n_dims)[:self.k]

    # xs_b: (self.b_size, n_points, self.n_dims)
    def evaluate(self, xs_b):
        assert xs_b.shape[2] >= self.k

        ys_b = torch.ones(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for idx in self.indices:
            ys_b *= xs_b[:, :, idx]
        return ys_b
        
    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy