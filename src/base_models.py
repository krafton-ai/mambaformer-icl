import math
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, in_size=50, hidden_size=1000, out_size=1):
        super(NeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ParallelNetworks(nn.Module):
    def __init__(self, num_models, model_class, **model_class_init_args):
        super(ParallelNetworks, self).__init__()
        self.nets = nn.ModuleList(
            [model_class(**model_class_init_args) for i in range(num_models)]
        )

    def forward(self, xs):
        assert xs.shape[0] == len(self.nets)

        for i in range(len(self.nets)):
            out = self.nets[i](xs[i])
            if i == 0:
                outs = torch.zeros(
                    [len(self.nets)] + list(out.shape), device=out.device
                )
            outs[i] = out
        return outs

# source: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L50-L57C119
class GELU_new(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class LowerTriLinear(nn.Linear):
    def __init__(self, in_sizes, out_size):
        super().__init__(in_sizes, out_size)
        with torch.no_grad():
            self.weight.copy_(torch.tril(self.weight))
        self.weight.register_hook(lambda grad: grad * torch.tril(torch.ones_like(grad)))

class CausalLinearAttention(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, dropout=0.1):
        super(CausalLinearAttention, self).__init__()

        self.n_head = n_head
        assert d_model % self.n_head == 0
        self.d_k = d_model // self.n_head

        # self.W = LowerTriLinear(max_seq_len, max_seq_len)
        # there shouldn't be input dependence - Q, K will not be multiplied with xs
        self.Q = nn.ModuleList([    
            nn.Linear(self.d_k, max_seq_len, bias=False)
        for i in range(self.n_head)
        ])
        self.K = nn.ModuleList([
            nn.Linear(max_seq_len, self.d_k, bias=False)
        for i in range(self.n_head)
        ])

        # use the same d_k and d_v
        self.V = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False)
        for i in range(self.n_head)
        ])
        
        self.W1 = nn.Linear(d_model, 4*d_model)
        self.W2 = nn.Linear(4*d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)

        self.activation = GELU_new()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    # xs.shape: [b_size, seq_len, d_model]
    # xs = inputs_embeds
    def forward(self, xs):
        seq_len = xs.shape[1]
        W = [
            torch.matmul(Q_i.weight, K_i.weight) 
                for Q_i, K_i in zip(self.Q, self.K)
        ]
        W = [torch.tril(W_i[:seq_len, :seq_len]) for W_i in W]
        V = [V_i(xs) for V_i in self.V]
        # attention equivalent
        # xs_p == z for single head; xs_p == z_i for multi-head
        xs_p = [torch.matmul(W_i, V_i) for W_i, V_i in zip(W, V)]
        xs_p = self.Wo(torch.cat(xs_p, dim=2))
        xs_p = self.dropout(self.norm1(xs_p))
        xs_p += xs
        xs_pp = self.W2(self.dropout(self.activation(self.W1(xs_p))))
        return self.norm2(xs_pp)+xs_p