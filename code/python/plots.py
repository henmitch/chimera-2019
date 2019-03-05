import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from run import order, chimera

size = (8, 3)

parser = argparse.ArgumentParser(
    description="Plot the results of the chimera experiments")
parser.add_argument(
    "f", metavar="f", type=str, nargs=1,
    help="The file of which we're making a plot."
)

file = parser.parse_args().f[0]

with open(file, "rb") as f:
    [(b, i0, x_rev, λ, θ, μ, s, x_rest, α, n1, β, n2, G1, G2),
     sol, phase, χ, m] = pickle.load(f)

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

χ = chimera(phase, cortices)

ρ_bar = np.mean(np.array([order(phase[:, low:high])
                          for [low, high] in cortices]),
                axis=0)

chi = np.sum(np.array([(order(phase[:, low:high]) - ρ_bar)**2
                       for [low, high] in cortices]), axis=0)/len(cortices)

title = f"alpha: {α:.03f}, beta: {β:.03f}, chi: {χ:.04f}"

plt.matshow(phase, aspect="auto")
plt.title(title)
plt.colorbar()
plt.savefig(f"overhead-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

plt.plot(np.cos(np.mean(phase, axis=1)))
plt.title(title)
plt.xlabel("time")
plt.ylabel("cos(mean(phase))")
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"cos_mean-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

for cortex in cortices:
    [low, high] = cortex
    plt.plot(np.cos(np.mean(phase[:, low:high], axis=1)),
             label=f"{low + 1}-{high}", lw=0.05)
plt.xlabel("time")
plt.ylabel("cos(mean(phi))")
plt.legend(bbox_to_anchor=(1, 1))
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"by_cortex_cos_mean-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

plt.plot(order(phase))
plt.xlabel("time")
plt.ylabel("rho")
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"order-{α:.03f}-{β:.03f}.png", dpi=500,
            bbox_inches="tight", format="png")
plt.cla()
plt.clf()

for cortex in cortices:
    [low, high] = cortex
    plt.plot((order(phase[:, low:high]) - ρ_bar)**2,
             label=f"{low + 1}-{high}", lw=0.05)
plt.xlabel("time")
plt.ylabel("(rho - rho bar)**2")
plt.legend(bbox_to_anchor=(1, 1))
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"by_cortex_variance-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

plt.plot(chi)
plt.xlabel("time")
plt.ylabel("sigma(chi(t))")
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"chi-{α:.03f}-{β:.03f}.png", dpi=500,
            bbox_inches="tight", format="png")
plt.cla()
plt.clf()
