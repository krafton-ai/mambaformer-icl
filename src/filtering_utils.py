import torch

def zero_pad_ys(tokenized, xs, ys):
    if tokenized:
        (bsize, points), dim = xs.shape, 1
    else:
        bsize, points, dim = xs.shape
    
    ys = torch.cat(
        (
            ys.view(bsize, points, 1),
            torch.zeros(bsize, points, dim - 1, device=ys.device),
        ),
        axis=2,
    )
    return ys

def orthogonalize(vector, basis):
    # Compute dot products in batch
    dot_vector_basis = torch.sum(vector * basis, dim=1, keepdim=True)
    dot_basis_basis = torch.sum(basis * basis, dim=1, keepdim=True)

    # Perform the orthogonalization in batch
    return vector - (basis * dot_vector_basis / dot_basis_basis)

def generate_dummy_vector(w):
    u = orthogonalize(torch.randn(w.shape, device=w.device), w)
    v = orthogonalize(torch.randn(w.shape, device=w.device), w)
    
    # make u \perp v
    v = orthogonalize(v, u)

    a = torch.randn(u.shape[0], device=w.device)
    b = torch.randn(u.shape[0], device=w.device)

    # if a**2+b**2 == 1
    # std(vec) == \sqrt(d)
    a = a / (a**2+b**2)
    b = b / (a**2+b**2)

    vec = a.unsqueeze(1)*u.squeeze(-1) + b.unsqueeze(1)*v.squeeze(-1)
    #     a = a / torch.sqrt(a**2+b**2)
    #     b = b / torch.sqrt(a**2+b**2)

    #     vec = torch.sqrt(torch.tensor(v.shape[1])) * (a.unsqueeze(1)*u.squeeze(-1) + b.unsqueeze(1)*v.squeeze(-1))


    return vec
    

def generate_dummy_vectors(task, task_name, vector, dummy_ind=None):
    if task_name == "filter_linear_regression" or task_name == "filter_relu_2nn_regression":
        if len(dummy_ind.shape) == 1:
            if len(vector.shape) == 2:
                vector[:, dummy_ind] = torch.ones_like(vector[:, dummy_ind])
            else:
                vector[:, dummy_ind, :] = torch.ones_like(vector[:, dummy_ind, :])
        else:
            vector[dummy_ind] = torch.ones_like(vector[dummy_ind])
        return vector
    elif task_name == "filter_scale_linear_regression":
        assert dummy_ind is not None
        # eval
        if len(dummy_ind.shape) == 1:
            vector[:, dummy_ind, :] = vector[:, dummy_ind, :]*task.dummy_scale
            return vector
        else:
            scale_tensor = torch.where(dummy_ind, task.dummy_scale, torch.tensor(1.0))
            scale_tensor = scale_tensor.unsqueeze(-1).expand_as(vector).to(vector.device)
            return vector * scale_tensor
    elif task_name == "filter_ortho_linear_regression":
        w = task.w_b.to(vector.device)
        dummy_vectors = generate_dummy_vector(w)
        expanded_dummy_vec = dummy_vectors.unsqueeze(1).expand_as(vector)
        expanded_inds = dummy_ind.unsqueeze(-1).expand_as(vector)
        X = torch.where(expanded_inds, expanded_dummy_vec, vector)
        return torch.where(expanded_inds, expanded_dummy_vec, vector)

    else:
        raise NotImplementedError("Invalid filtering task name")

def dummy_replace(filtering_inds, task, task_name, vector):
    replaced_vector = generate_dummy_vectors(task, task_name, vector, filtering_inds)
    return replaced_vector

if __name__ == "__main__":
    u = torch.randn(20)
    v = torch.randn(20)

    u = orthogonalize(u, v)

    print(torch.dot(u, v))
