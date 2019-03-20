import gc
import numpy as np
import os
import pandas as pd
import pickle

from run import chimera

out = pd.DataFrame(columns=["alpha", "beta", "metastability", "chimera"])

tmax = 4000
N = 100*tmax

metadata = pd.read_excel("../connectomes/mouse_meta.xlsx", sheet_name=None)

m = metadata["Voxel Count_295 Structures"]
m = m.loc[m["Represented in Linear Model Matrix"] == "Yes"]

columns = []
cortices = [[0, 0]]
for region in m["Major Region"].unique():
    i = [columns.append(acronym.replace(" ", "")) for acronym in
         m.loc[m["Major Region"] == region, "Acronym"].values]
    cortices.append([cortices[-1][-1], cortices[-1][-1] + len(i)])
cortices.remove([0, 0])
del(metadata)
del(m)

for i in os.listdir("/users/h/m/hmmitche/thesis/data"):
    print(i)
    with open(f"/users/h/m/hmmitche/thesis/data/{i}", "rb") as f:
        params, sol, phase, χ, m = pickle.load(f)
    χ = chimera(phase, cortices)
    α, β = params[8], params[10]
    vals = sol.y.T.reshape(N, 3, -1)
    end = {"alpha": α,
           "beta": β,
           "max_phase": phase.max(),
           "chimera": χ,
           "metastability": m}
    out = out.append(end, ignore_index=True)
    with open(f"/users/h/m/hmmitche/thesis/data/{i}", "wb") as f:
        pickle.dump([params, sol, phase, χ, m], f)
    gc.collect()


with open("../../data/hizanidis_params.pkl", "wb") as f:
    pickle.dump(out, f)
