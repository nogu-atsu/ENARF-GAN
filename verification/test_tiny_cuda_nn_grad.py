import tinycudann as tcnn
import torch
from torch import nn
import time
from torch.cuda.amp import autocast


def run(n_input_dims):
    print("n_input_dims:", n_input_dims)
    network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2
    }
    encoding_config = {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16
    }
    n_output_dims = 4

    # define tiny-cuda-nn model
    ingp = tcnn.NetworkWithInputEncoding(
        n_input_dims, n_output_dims,
        encoding_config, network_config
    )
    encoding = tcnn.Encoding(n_input_dims, encoding_config)
    # measure time
    n_loop = 100

    # x = torch.randn(int(2 ** 2), 3, device="cuda").requires_grad_(True) * 10000
    x = -torch.ones(1, 3, device="cuda")
    emb = encoding(x)
    print(emb)

    x = -torch.ones(1, 3, device="cuda") * 0.5
    emb = encoding(x)
    print(emb)

    # with torch.set_grad_enabled(False):
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     for i in range(n_loop):
    #         out = ingp(x)
    #         # out.sum().backward()
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     print(f"ffmlp time:", end - start)

    # print(x.grad.std())


if __name__ == "__main__":
    run(n_input_dims=3)
