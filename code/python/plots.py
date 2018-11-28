import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from pympler import muppy, summary

tmax = 4000
N = 100*tmax
t = np.linspace(0, tmax, N)

for i in os.listdir("/gpfs2/scratch/hmmitche/mouse_data/"):
    print(i, end="... ")
    with open(f"/gpfs2/scratch/hmmitche/mouse_data/{i}", "rb") as f:
        data = pickle.load(f)
    params = data[0]
    α, β = params[8], params[10]
    if f"{α:0.3f}-{β:0.3f}.png" in os.listdir("/users/h/m/hmmitche/thesis/figure/"):
        print("Already exists")

    else:
        sum1 = summary.summarize(muppy.get_objects())
        vals = data[1]
        del(data)
        vals = vals.sol(t).T.reshape(N, 3, -1)
        fig, ax = plt.subplots(1, 1)

        plt.matshow(vals[:, 0, :], aspect="auto")
        plt.title(f"{α:0.3f}-{β:0.3f}")
        plt.savefig(f"/users/h/m/hmmitche/thesis/figure/{α:0.3f}-{β:0.3f}.png", format="png")
        plt.close(fig)
        ax.cla()
        fig.clf()
        plt.close(fig)
        del(vals)
        del(fig)
        del(ax)

        gc.collect()
        print("Picture made")
        sum2 = summary.summarize(muppy.get_objects())
        summary.print_(summary.get_diff(sum1, sum2))
