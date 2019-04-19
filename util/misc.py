import pathlib
import random
from time import time

import torch


def fix_seed():
    random.seed(0)
    torch.manual_seed(0)


def get_path(args):
    path = pathlib.Path("checkpoints") / args.dataset / args.token_type / args.loss
    if args.loss != "xent":
        path /= args.emb_type
    if args.tied:
        path /= "tied"
    if args.loss in ["vmfapprox_paper", "vmfapprox_fixed", "vmf"]:
        path /= f"reg1{args.reg_1}_reg2{args.reg_2}"
    return path


class TimeMeter:
    def __init__(self):
        self.start = time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    def avg(self):
        return self.n / self.elapsed_time()

    def elapsed_time(self):
        return time() - self.start


class StopwatchMeter:
    def __init__(self, sum_=0):
        self.sum = sum_
        self.start_time = None

    def start(self):
        self.start_time = time()

    def stop(self):
        if self.start_time is not None:
            delta = time() - self.start_time
            self.sum += delta
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.start_time = None
