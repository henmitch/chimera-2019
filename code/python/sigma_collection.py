import gc
import numpy as np
import os
import pandas as pd
import pickle

def order(phases):
    return np.abs(np.sum(np.exp(phases*1j), axis=1)/phases.shape[1])

def metastability(phase, cortices, p, channel=0):
    metastabilities = []
    N = int((1-p)*phase.shape[0])
    for cortex in cortices:
        ph = phase[N:, cortex[0]:cortex[1]]
        metastabilities.append(np.sum((order(ph) - np.mean(order(ph)))**2)/(N - 1))
    return np.mean(metastabilities)

def chimera(phase, cortices, p, channel=0):
    N = int((1-p)*phase.shape[0])
    chimeras = []
    M = len(cortices)
    average = np.mean([order(phase[N:, cortex[0]:cortex[1]]) for cortex in cortices])
    s = np.zeros(phase.shape[0] - N)
    for cortex in cortices:
        ph = phase[N:, cortex[0]:cortex[1]]
        s += (order(ph) - average)**2
    return np.mean(s/(M - 1))

out = pd.DataFrame(columns=["alpha", "beta", "metastability", "chimera"])

tmax = 4000
N = 100*tmax
t = np.linspace(0, tmax, N)

cortices = [[0, 38],
            [38, 73],
            [73, 98],
            [98, 119],
            [119, 139],
            [139, 152],
            [152, 164],
            [164, 175],
            [175, 186],
            [186, 195],
            [195, 203],
            [203, 210],
            [210, 213]]

for i in os.listdir("/users/h/m/hmmitche/thesis/data"):
    with open(f"/users/h/m/hmmitche/thesis/data/{i}", "rb") as f:
        data = pickle.load(f)
    params = data[0]
    α, β = params[8], params[10]
    ncs, sig = data[-2], data[-1]
    vals = data[1].sol(t).T.reshape(N, 3, -1)
    phase = data[2]
    end = {"alpha": α,
           "beta": β,
           "metastability": metastability(phase, cortices, 0.2),
           "chimera": chimera(phase, cortices, 0.2)}
    out = out.append(end, ignore_index=True)
    gc.collect()


with open("../../data/hizandis_params.pkl", "wb") as f:
    pickle.dump(out, f)
