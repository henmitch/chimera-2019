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
plt.savefig(f"../figure/overhead-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

plt.plot(np.cos(np.mean(phase, axis=1)))
plt.title(title)
plt.xlim([0, phase.shape[0]])
plt.ylim([-1.05, 1.05])
plt.xlabel("time")
plt.ylabel("cos(mean(phase))")
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"../figure/cos_mean-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

for cortex in cortices:
    [low, high] = cortex
    plt.plot(np.cos(np.mean(phase[:, low:high], axis=1)),
             label=f"{low + 1}-{high}", lw=0.05)
plt.xlim([0, phase.shape[0]])
plt.ylim([-1.05, 1.05])
plt.xlabel("time")
plt.ylabel("cos(mean(phi))")
plt.legend(bbox_to_anchor=(1, 1))
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"../figure/by_cortex_cos_mean-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

plt.plot(order(phase))
plt.xlim([0, phase.shape[0]])
plt.xlabel("time")
plt.ylabel("rho")
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"../figure/order-{α:.03f}-{β:.03f}.png", dpi=500,
            bbox_inches="tight", format="png")
plt.cla()
plt.clf()

for cortex in cortices:
    [low, high] = cortex
    plt.plot((order(phase[:, low:high]) - ρ_bar)**2,
             label=f"{low + 1}-{high}", lw=0.05)
plt.xlim([0, phase.shape[0]])
plt.xlabel("time")
plt.ylabel("(rho - rho bar)**2")
plt.legend(bbox_to_anchor=(1, 1))
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"../figure/by_cortex_variance-{α:.03f}-{β:.03f}.png",
            dpi=500, bbox_inches="tight", format="png")
plt.cla()
plt.clf()

plt.plot(chi)
plt.xlim([0, phase.shape[0]])
plt.xlabel("time")
plt.ylabel("sigma(chi(t))")
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"../figure/chi-{α:.03f}-{β:.03f}.png", dpi=500,
            bbox_inches="tight", format="png")
plt.cla()
plt.clf()

fig, [overhead_ax,
      order_ax,
      cos_mean_ax,
      by_cortex_cos_mean_ax,
      variance_ax,
      by_cortex_variance_ax] = plt.subplots(6, 1, sharex=True, squeeze=True)
overhead_ax.set_xlim([0, phase.shape[0]])

overhead_ax.matshow(phase.T, aspect="auto", origin="lower")

cos_mean_ax.plot(np.cos(np.mean(phase, axis=1)), lw=0.1)

for cortex in cortices:
    [low, high] = cortex
    by_cortex_cos_mean_ax.plot(np.cos(np.mean(phase[:, low:high], axis=1)),
                            label=f"{low + 1}-{high}", lw=0.07)

order_ax.plot(order(phase), lw=0.1)
order_ax.set_ylim([])

variance_ax.plot(chi, lw=0.1)

for cortex in cortices:
    [low, high] = cortex
    by_cortex_variance_ax.plot(((order(phase[:, low:high]) - ρ_bar)**2),
                            label=f"{low + 1}-{high}", lw=0.07)

fig.set_size_inches(*size, forward=True)
plt.savefig(f"../figure/all-{α:.03f}-{β:.03f}.png", dpi=500,
            bbox_inches="tight", format="png")
plt.cla()
plt.clf()