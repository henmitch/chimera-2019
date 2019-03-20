import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from matplotlib import cm
from run import order, chimera


current_cmap = cm.get_cmap()
current_cmap.set_bad(color='black')

size = (12, 6)


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

mtdt = metadata["Voxel Count_295 Structures"]
del(metadata)
mtdt = mtdt.loc[mtdt["Represented in Linear Model Matrix"] == "Yes"]

columns = []
cortices = [[0, 0]]
for region in mtdt["Major Region"].unique():
    i = [columns.append(acronym.replace(" ", "")) for acronym in
         mtdt.loc[mtdt["Major Region"] == region, "Acronym"].values]
    cortices.append([cortices[-1][-1], cortices[-1][-1] + len(i)])
cortices.remove([0, 0])
del(mtdt)


ρ_bar = np.mean(np.array([order(phase[:, low:high])
                          for [low, high] in cortices]),
                axis=0)

chi = np.sum(np.array([(order(phase[:, low:high]) - ρ_bar)**2
                       for [low, high] in cortices]), axis=0)/len(cortices)


title = f"alpha: {α:.03f}, beta: {β:.03f}, chimera: {χ:.04f}, metastability: {m:.04f}"


sim = np.empty([phase.shape[1] + 2*(len(cortices) - 1), phase.shape[0]])
sim[:] = np.nan


for i, cortex in enumerate(cortices):
    sim[(cortex[0] + 2*i):(cortex[1] + 2*i)] = phase.T[cortex[0]:cortex[1]]

plt.imshow(sim, aspect="auto", origin="lower", interpolation="none")
[i.set_linewidth(0.01) for i in plt.gca().spines.values()]
plt.xlim([0, phase.shape[0]])
plt.title(title)
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
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
plt.ylim([-0.025, 1.025])
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
plt.ylim([-0.05, 0.62])
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
plt.ylim([-0.01, 0.15])
plt.xlabel("time")
plt.ylabel("sigma(chi(t))")
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(*size, forward=True)
plt.savefig(f"../figure/chi-{α:.03f}-{β:.03f}.png", dpi=500,
            bbox_inches="tight", format="png")
plt.cla()
plt.clf()


xticks = [cortex[1] + 2*i + 1 for i, cortex in enumerate(cortices)][:-1]


fig, [overhead_ax,
      order_ax,
      cos_mean_ax,
      by_cortex_cos_mean_ax,
      variance_ax,
      by_cortex_variance_ax] = plt.subplots(6, 1, sharex=True, squeeze=True)

overhead_ax.set_title(title)

[i.set_linewidth(0.01) for i in overhead_ax.spines.values()]
overhead_ax.set_xlim([0, phase.shape[0]])

overhead_ax.matshow(sim, aspect="auto", origin="lower", interpolation="none")
overhead_ax.set_yticks(xticks)
overhead_ax.set_yticklabels(["" for i in xticks])
overhead_ax.tick_params(axis="y",
                        width=0.2)

order_ax.plot(order(phase), lw=0.1)
order_ax.set_ylim([-0.025, 1.025])

cos_mean_ax.plot(np.cos(np.mean(phase, axis=1)), lw=0.1)
cos_mean_ax.set_ylim([-1.05, 1.05])

for cortex in cortices:
    [low, high] = cortex
    by_cortex_cos_mean_ax.plot(np.cos(np.mean(phase[:, low:high], axis=1)),
                               label=f"{low + 1}-{high}", lw=0.07)
by_cortex_cos_mean_ax.set_ylim([-1.05, 1.05])

variance_ax.plot(chi, lw=0.1)
variance_ax.set_ylim([-0.01, 0.15])

for cortex in cortices:
    [low, high] = cortex
    by_cortex_variance_ax.plot(((order(phase[:, low:high]) - ρ_bar)**2),
                               label=f"{low + 1}-{high}", lw=0.07)
by_cortex_variance_ax.set_ylim([-0.05, 0.62])

fig.set_size_inches(*size, forward=True)
fig.savefig(f"../figure/all-{α:.03f}-{β:.03f}.png", dpi=700,
            bbox_inches="tight", format="png")
plt.cla()
plt.clf()
