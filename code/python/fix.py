import argparse
import pandas as pd
import pickle
import run

metadata = pd.read_excel("../connectomes/mouse_meta.xlsx", sheet_name=None)

m = metadata["Voxel Count_295 Structures"]
del(metadata)
m = m.loc[m["Represented in Linear Model Matrix"] == "Yes"]

columns = []
cortices = [[0, 0]]
for region in m["Major Region"].unique():
    i = [columns.append(acronym.replace(" ", "")) for acronym in
         m.loc[m["Major Region"] == region, "Acronym"].values]
    cortices.append([cortices[-1][-1], cortices[-1][-1] + len(i)])
cortices.remove([0, 0])

parser = argparse.ArgumentParser(
    description="Fix the chimera results of the input files"
)
parser.add_argument(
    "f", metavar="f", type=str, nargs=1,
    help="The file whose chimera value we're fixing"
)

file = parser.parse_args().f[0]

with open(file, "rb") as f:
    [params, sol, phase, χ, m] = pickle.load(f)

χ = run.chimera(phase, cortices)

with open(file, "wb") as f:
    pickle.dump([params, sol, phase, χ, m], f)
