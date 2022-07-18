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

    n_output_dims = 4
    n_neurons = network_config["n_neurons"]
    n_hidden_layers = network_config["n_hidden_layers"]

    # define tiny-cuda-nn model
    ffmlp = tcnn.Network(n_input_dims, n_output_dims, network_config).to("cuda")

    # define pytorch mlp
    layers = [nn.Linear(n_input_dims, n_neurons), nn.ReLU()]
    for i in range(n_hidden_layers):
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(n_neurons, n_output_dims))
    mlp = nn.Sequential(*layers).to("cuda")

    ffmlp.eval()
    mlp.eval()

    # measure time
    n_loop = 100

    x = torch.randn(int(2 ** 21), n_input_dims, device="cuda")

    with torch.set_grad_enabled(False):
        torch.cuda.synchronize()
        start = time.time()
        for i in range(n_loop):
            ffmlp_out = ffmlp(x)
        torch.cuda.synchronize()
        end = time.time()
        print(f"ffmlp time:", end - start)

        with autocast():
            torch.cuda.synchronize()
            start = time.time()
            for i in range(n_loop):
                mlp_out = mlp(x)
            torch.cuda.synchronize()
            end = time.time()
            print(f"torch mlp time:", end - start)


if __name__ == "__main__":
    run(n_input_dims=8)
    run(n_input_dims=16)
    run(n_input_dims=17)
    run(n_input_dims=32)
    run(n_input_dims=64)
    run(n_input_dims=128)
