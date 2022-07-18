import torch
import torch.nn.functional as F
import numpy as np


def sample_unit_vectors_uniformly(n_samples, n_dim):
    """
    Sample n_samples unit vectors uniformly.
    """
    out = torch.abs(torch.randn(n_samples, n_dim, device='cuda'))
    out = F.normalize(out, p=2, dim=1)
    return out


code_book = sample_unit_vectors_uniformly(65536, 8)
vec = sample_unit_vectors_uniformly(1000, 8)

idx1 = [torch.mm(code_book, vec[None, i].t()).argmax(dim=0) for i in range(vec.shape[0])]
dot = torch.sum(vec * code_book[idx1], dim=1)
norm = torch.norm(code_book[idx1] - vec, p=2, dim=1)
print(dot.min(), dot.mean())
print(norm.max(), norm.mean())
# residual = F.normalize(vec - code_book[idx1], dim=1)
# norm = torch.norm(vec - code_book[idx1], dim=1)
# idx2 = torch.mm(code_book, residual.t()).argmax(dim=0)
#
# approx = code_book[idx1] + code_book[idx2] * norm[:, None]
# print(torch.norm(approx, dim=1).min(), torch.norm(approx, dim=1).mean())
#
# dot = torch.sum(vec * approx, dim=1)
# print(dot.min(), dot.mean())


# idx = torch.topk(torch.mm(code_book, vec.t()), k=10, dim=0)[1]
# approx = code_book[idx]  # 10, 1000, 32
