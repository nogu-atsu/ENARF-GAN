import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(".")
from cuda_extension.triplane_sampler import grid_sampler

a = torch.empty((3, 32, 256, 256), dtype=torch.float32, device="cuda").uniform_().requires_grad_(True)
index = torch.empty((3, 320, 1000, 2), dtype=torch.float32, device="cuda").uniform_(-1, 1).requires_grad_(False)

n_loop = 100

# pytorch version
torch.cuda.synchronize()
s = time.time()
for i in range(n_loop):
    torch.cuda.synchronize()
    out1 = F.grid_sample(a, index, mode="bilinear", align_corners=False)
    out1.sum().backward()
torch.cuda.synchronize()
print("case 1:", time.time() - s)
pytorch_grad_a = a.grad.clone()
# pytorch_grad_index = index.grad.clone()

a.grad = None
index.grad = None

# cpp version
Bilinear = 0
Zeros = 0
torch.cuda.synchronize()
s = time.time()
for i in range(n_loop):
    torch.cuda.synchronize()
    out1 = grid_sampler(a, index, mode="bilinear", align_corners=False)
    out1.sum().backward()
torch.cuda.synchronize()
print("cpp:", time.time() - s)
cpp_grad_a = a.grad
cpp_grad_index = index.grad

# print(torch.isclose(pytorch_grad_a, cpp_grad_a).all())
# print(torch.isclose(pytorch_grad_index, cpp_grad_index).all())

# print(cpp_grad_a)
print(cpp_grad_index)