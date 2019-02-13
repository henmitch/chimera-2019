import gc
import numpy as np
import os
import pandas as pd
import pickle

out = pd.DataFrame(columns=["alpha", "beta", "metastability", "chimera"])

tmax = 4000
N = 100*tmax

for i in os.listdir("/users/h/m/hmmitche/thesis/data"):
    print(i)
    with open(f"/users/h/m/hmmitche/thesis/data/{i}", "rb") as f:
        params, sol, phase, χ, m = pickle.load(f)
        α, β = params[8], params[10]
        vals = sol.y.T.reshape(N, 3, -1)
        end = {"alpha": α,
               "beta": β,
               "chimera": χ,
               "metastability": m}
        out = out.append(end, ignore_index=True)
        gc.collect()


with open("../../data/hizanidis_params.pkl", "wb") as f:
    pickle.dump(out, f)
