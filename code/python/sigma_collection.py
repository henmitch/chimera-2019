import os
import pandas as pd
import pickle

out = pd.DataFrame(columns=["alpha", "beta", "sigma", "ncs"])

for i in os.listdir("../../data/"):
    with open(f"../../data/{i}", "rb") as f:
        data = pickle.load(f)
    params = data[0]
    α, β = params[8], params[10]
    ncs, sig = data[-2], data[-1]
    end = {"alpha": α, "beta": β, "sigma": sig, "ncs": ncs}
    out = out.append(end, ignore_index=True)

with open("test.pkl", "wb") as f:
    pickle.dump(out, f)
