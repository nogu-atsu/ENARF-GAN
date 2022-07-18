# grid_sampleで，テンソルを細切れにしてgrid_sampleを繰り返すのと，一気にgrid_sampleするのでどちらが速いかを比較

# ssoの実験だと，grid_sampleでは512 ** 2 * 112 = 29360128個の点からのサンプリングを行っている．
# パーツの個数で数倍，人の領域で数分の一になることを考慮すると，概ね3200万点のサンプリングを行っている．

# 試すこと
# fp16

import torch
import torch.nn.functional as F
import time
import triplane_sampler_cuda

a = torch.empty((3, 32, 256, 256), dtype=torch.float32, device="cuda").uniform_()
index = torch.empty((3, 32000, 1000, 2), dtype=torch.float32, device="cuda").uniform_(-1, 1)
# index = torch.empty((3, 1, 32000000, 2), dtype=torch.float32, device="cuda").uniform_(-0.2, 0.2)
# index = torch.linspace(-1, 1, 32000000, device="cuda")[None, None, :, None].repeat(3, 1, 1, 2)

n_loop = 100
# # case 2: 細切れにsample
# torch.cuda.synchronize()
# s = time.time()
# out2 = 0
# for i in range(a.shape[0]):
#     out2 += F.grid_sample(a[None, i], index[None, i])
# torch.cuda.synchronize()
# print("case 2:", time.time() - s)
#
# # case 3: 細切れにsample
# torch.cuda.synchronize()
# s = time.time()
# out2 = 0
# for i in range(a.shape[1]):
#     out2 += F.grid_sample(a[:, None, i], index)
# torch.cuda.synchronize()
# print("case 3:", time.time() - s)
#
# # case 4: 細切れにsample
# torch.cuda.synchronize()
# s = time.time()
# out2 = 0
# interval = 1
# for i in range(0, a.shape[1], interval):
#     out2 += F.grid_sample(a[:, i:i + interval], index)
# torch.cuda.synchronize()
# print("case 4:", time.time() - s)

# case 1: 一気にサンプル
torch.cuda.synchronize()
s = time.time()
for i in range(n_loop):
    # torch.cuda.synchronize()
    out1 = F.grid_sample(a, index, mode="bilinear", align_corners=False)  # (3, 32, 32000, 1000)
torch.cuda.synchronize()
print("case 1:", time.time() - s)

# # case 5: 細切れにsample
# torch.cuda.synchronize()
# s = time.time()
# out2 = []
# for j in range(0, index.shape[1], 4000):
#     out2.append(F.grid_sample(a, index[:, j:j + 4000]))
# out2 = torch.cat(out2)
# torch.cuda.synchronize()
# print("case 5:", time.time() - s)

Bilinear = 0
Zeros = 0
torch.cuda.synchronize()
s = time.time()
for i in range(n_loop):
    # torch.cuda.synchronize()
    out1 = triplane_sampler_cuda.forward(a, index, Bilinear, Zeros, False)  # (3, 32, 32000, 1000)
torch.cuda.synchronize()
print("cpp:", time.time() - s)

# print(torch.isclose(out1, out2).all())