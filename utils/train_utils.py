import argparse
import os
import subprocess
import sys

import torch
import yaml
from easydict import EasyDict as edict


def record_setting(out):
    """Record scripts and commandline arguments"""
    # out = out.split()[0].strip()
    source = out + "/source"
    if not os.path.exists(source):
        os.system('mkdir -p %s' % source)
        # os.mkdir(out)

    # subprocess.call("cp *.py %s" % source, shell=True)
    # subprocess.call("cp configs/*.yml %s" % out, shell=True)

    subprocess.call("find . -type d -name result -prune -o -name '*.py' -print0"
                    "| xargs -0 cp --parents -p -t %s" % source, shell=True)
    subprocess.call("find . -type d -name result -prune -o -name '*.yml' -print0|"
                    " xargs -0 cp --parents -p -t %s" % source, shell=True)

    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")


def write(iter, loss, name, writer):
    writer.add_scalar("metrics/" + name, loss, iter)
    return loss


def all_reduce(scalar):
    scalar = torch.tensor(scalar).cuda()
    torch.distributed.all_reduce(scalar)
    return scalar.item()