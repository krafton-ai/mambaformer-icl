import math
import random
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.inds = None

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "gaussian_for_retrieval": GaussianRetrievalSampler,
        "sparse_parity_sampler": SparseParitySampler,
        "unique_key_tokens_fixed_query": FixedQueryTokenSampler,
        "unique_key_tokens_random_query": RandomQueryTokenSampler,
        "filter": FilteringGaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        if "token" in data_name:
            return sampler_cls(vocab_size=kwargs.get("vocab_size"))
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

# TODO: Do not generate all vocab_size
def randperm_2d_with_query(bsize, vocab_size, n_points, query_pos=None, query_vector=None):
    if query_vector is None:
        query_vector = torch.zeros((bsize, 1), dtype=torch.int64, device="cuda")
    elif len(query_vector.shape) == 1:
        query_vector = query_vector.unsqueeze(-1)
    assert query_vector.shape[0] == bsize

    # Shuffle each row according to the generated indices
    keys = torch.arange(vocab_size-1, device="cuda").repeat(bsize, 1)
    keys += (keys >= query_vector).int()  # make space for unique query key
    indices = torch.stack([torch.randperm(vocab_size-1, device="cuda") for _ in range(bsize)])
    keys = keys.gather(1, indices)[:, :n_points-1]

    if query_pos is None:
        keys = torch.cat((keys, query_vector), dim=1)
        indices = torch.stack([torch.randperm(n_points, device="cuda") for _ in range(bsize)])
        return keys.gather(1, indices)

    left = tensor[:, :query_pos]  # Tensor slice before the index
    right = tensor[:, query_pos:] # Tensor slice after the index
    return torch.cat((left, query_vector, right), dim=1)      


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims, device="cuda")
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, device="cuda")
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator, device="cuda")
        if self.scale is not None:
            xs_b = xs_b @ self.scale.cuda()
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class GaussianRetrievalSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(
        self,
        n_points,
        b_size,
        n_dims_truncated=None,
        seeds=None,
        query_pos=None,
        noise_sigma=0.0,
    ):
        is_random_query = (query_pos is None) # else, should choose pair at index:query_pos
        if seeds is None:
            xs_b = torch.randn(b_size, 2 * n_points + 1, self.n_dims, device="cuda")
            for i in range(b_size):
                if is_random_query:
                    query_pos = 2 * random.randint(0, n_points)
                xs_b[i][-1, :] = xs_b[i][query_pos, :] 
        else:
            xs_b = torch.zeros(b_size, 2 * n_points + 1, self.n_dims, device="cuda")
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                random.seed(seed)
                xs_b[i] = torch.randn(2 * n_points + 1, self.n_dims, generator=generator, device="cuda")
                if is_random_query:
                    query_pos = 2 * random.randint(0, n_points)
                xs_b[i][-1, :] = xs_b[i][query_pos, :]
        # Normalize to sqrt(d)-ball
        xs_b = xs_b / torch.norm(xs_b, p=2, dim=-1, keepdim=True) * math.sqrt(self.n_dims)
        if noise_sigma > 0:
            for i in range(b_size):
                xs_b[i][-1, :] += noise_sigma * torch.randn(self.n_dims, device="cuda")
            xs_b = xs_b / torch.norm(xs_b, p=2, dim=-1, keepdim=True) * math.sqrt(self.n_dims)
        if self.scale is not None:
            xs_b = xs_b @ self.scale.cuda()
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class UniqueKeyTokenSampler(DataSampler):
    def __init__(self, vocab_size):
        assert vocab_size > 0
        super().__init__(n_dims=1)
        self.vocab_size = vocab_size

    def _sample_xs(
        self,
        n_points,
        b_size,
        n_dims_truncated=None,
        seeds=None,
        query_pos=None,
        fixed_query=True,
    ):
        assert self.vocab_size >= n_points  # keys should be unique
        is_random_query = (query_pos is None) # else, should choose pair at index:query_pos

        if seeds is None:
            if fixed_query:
                query_vector = torch.zeros((b_size, 1), dtype=torch.int64, device="cuda") 
            else:
                query_vector = torch.randint(0, self.vocab_size, (b_size, 1), device="cuda")
            keys = randperm_2d_with_query(b_size, self.vocab_size, n_points, query_pos, query_vector).unsqueeze(-1)
            values = torch.randint(1, self.vocab_size, (b_size, n_points, 1), device="cuda")

            # Interleave key and value
            xs_b = torch.cat((keys, values), dim=-1)
            xs_b = xs_b.reshape(b_size, 2*n_points)
            xs_b = torch.cat((xs_b, query_vector), dim=1)
        else:
            assert len(seeds) == b_size
            xs_b = torch.zeros((b_size, 2 * n_points + 1), dtype=torch.int64, device="cuda")
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                random.seed(seed)
                
                xs_b = torch.randint(1, self.vocab_size, (1, self.vocab_size, n_points), device="cuda")
                if fixed_query:
                    query_vector = torch.zeros((1, 1), dtype=torch.int64, device="cuda") 
                else:
                    query_vector = torch.randint(0, self.vocab_size, (1, 1), device="cuda")
                keys = randperm_2d_with_query(1, self.vocab_size, n_points, query_pos, query_vector).unsqueeze(-1)
                values = torch.randint(1, self.vocab_size, (1, n_points, 1), device="cuda")

                # Interleave key and value
                xs_b = torch.cat((keys, values), dim=-1).squeeze()
                xs_b = xs_b.reshape(2*n_points)
                xs_b = torch.cat((xs_b, query_vector[:, 0]))
        return xs_b


class FixedQueryTokenSampler(UniqueKeyTokenSampler):
    def __init__(self, vocab_size):
        super().__init__(vocab_size=vocab_size)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, query_pos=None):
        return self._sample_xs(n_points, b_size, n_dims_truncated, seeds, query_pos, fixed_query=True)


class RandomQueryTokenSampler(UniqueKeyTokenSampler):
    def __init__(self, vocab_size):
        super().__init__(vocab_size=vocab_size)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, query_pos=None):
        return self._sample_xs(n_points, b_size, n_dims_truncated, seeds, query_pos, fixed_query=False)


class SparseParitySampler(DataSampler):
    # if scale = 2, can test whether it can distinguish multiplication and addition?
    def __init__(self, n_dims, scale=None):
        super().__init__(n_dims)
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), device="cuda")
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, device="cuda")
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randint(0, 2, (n_points, self.n_dims), generator=generator, device="cuda")
        
        xs_b = xs_b * 2 - 1
        if self.scale is not None:
            xs_b = xs_b @ self.scale.cuda()
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class FilteringGaussianSampler(GaussianSampler):
    def __init__(self, n_dims, bias=None, scale=None, prob=0.0):
        super().__init__(n_dims)
        self.inds = None
        self.prob = prob

    # creates xs; xs.shape[:2] == ys.shape, pred.shape
    # xs.shape[2] == n_dim
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_b = super().sample_xs(n_points, b_size, n_dims_truncated, seeds)
        self.inds = torch.rand(xs_b.shape[0], xs_b.shape[1], device=xs_b.device) < self.prob
        return xs_b