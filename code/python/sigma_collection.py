import numpy as np
import os
import pandas as pd
import pickle

def order(phases):
    return np.abs(np.sum(np.exp(phases*1j), axis=1)/phases.shape[1])

def metastability(y, phase, p, channel=0):
    metastabilities = []
    N = y.shape[0]
    for cortex in cortices:
        v = y[int((1-p)*N):, channel, cortex[0]:cortex[1]]
        ph = phase[int((1-p)*N):, cortex[0]:cortex[1]]
        metastabilities.append(np.sum((order(ph) - np.mean(order(ph)))**2)/(int((1-p)*N) - 1))
    return np.mean(metastabilities)

out = pd.DataFrame(columns=["alpha", "beta", "sigma", "ncs"])

tmax = 4000
N = 100*tmax
t = np.linspace(0, tmax, N)

cortices = [[0, 18],
            [18, 28],
            [28, 46],
            [46, 65]]

for i in os.listdir("../../data/"):
    with open(f"../../data/{i}", "rb") as f:
        data = pickle.load(f)
    params = data[0]
    α, β = params[8], params[10]
    ncs, sig = data[-2], data[-1]
    vals = data[1].sol(t).T.reshape(-1, 3, 65)
    phase = data[2]
    end = {"alpha": α, "beta": β, "metastability": metastability(vals, phase, 0.5)}
    out = out.append(end, ignore_index=True)

with open("../../data/hizandis_params.pkl", "wb") as f:
    pickle.dump(out, f)
