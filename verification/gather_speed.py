# grid_sampleで，テンソルを細切れにしてgrid_sampleを繰り返すのと，一気にgrid_sampleするのでどちらが速いかを比較

# ssoの実験だと，grid_sampleでは512 ** 2 * 112 = 29360128個の点からのサンプリングを行っている．
# パーツの個数で数倍，人の領域で数分の一になることを考慮すると，概ね3200万点のサンプリングを行っている．

# 試すこと
# fp16

import torch
import torch.nn.functional as F
import time

a = torch.empty((64000, 32), dtype=torch.float32, device="cuda").uniform_()
index = torch.randint(0, 64000, (3 * 32000 * 1000 * 2, 1), dtype=torch.long, device="cuda")
# index = torch.empty((3, 1, 32000000, 2), dtype=torch.float32, device="cuda").uniform_(-0.2, 0.2)
# index = torch.linspace(-1, 1, 32000000, device="cuda")[None, None, :, None].repeat(3, 1, 1, 2)

# case 1: 一気にサンプル
torch.cuda.synchronize()
s = time.time()
out1 = torch.gather(a, 0, index.expand(-1, 32))
print(out1.shape)
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
