import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from scipy.integrate import solve_ivp


def prep_masks(n, cortices):
    cortex_mask = np.zeros_like(n)
    for i, cortex in enumerate(cortices):
        cortex_mask[cortex[0]:cortex[1], cortex[0]:cortex[1]] += i + 1
    G1 = n.copy()
    G1[cortex_mask == 0] = 0
    G2 = n.copy()
    G2[cortex_mask != 0] = 0
    events = [lambda t_in, y_in, i=i: event(t_in, y_in, i)
              for i in range(n.shape[0])]
    for e in events:
        e.direction = 1.0

    return G1, G2, events


def Θ(x, x_rev, λ, θ):
    xk, xj = np.meshgrid(x, x)
    return (xj - x_rev)/(1 + np.exp(-λ*(xk - θ)))


def dΘ_dx(x, λ, θ):
    final = np.ones((x.size, x.size))/(1 + np.exp(-λ*(x - θ)))
    np.fill_diagonal(final,
                     final.diagonal() +
                     x*λ*np.exp(-λ*(x - θ))/(1+np.exp(-λ*(x - θ)))**2)
    return final


def hr_dots(current, _, b, i0, x_rev, λ, θ, μ, s, x_rest,
            α, n1, β, n2, G1, G2):
    x, y, z = map(lambda k: k.flatten(), np.split(current, 3))
    theta = Θ(x, x_rev, λ, θ)
    dots = np.zeros_like(current).reshape(3, -1)
    dots[0] = y - (x**3) + b*(x**2) + i0 - z -\
        (α/n1)*np.sum(G1*theta, axis=1) - (β/n2)*np.sum(G2*theta, axis=1)
    dots[1] = 1 - 5*(x**2) - y
    dots[2] = μ*(s*(x - x_rest) - z)
    return np.hstack(dots)


def jac(_, y_in):
    x, y, z = map(lambda k: k.flatten(), np.split(y_in, 3))
    dtheta_dx = dΘ_dx(x, λ, θ)
    dẋ_dx = -3*x**2 + 2*b*x - (α/n1)*G1*dtheta_dx - (β/n2)*G2*dtheta_dx
    dẋ_dy = np.ones_like(dẋ_dx)
    dẋ_dz = -np.ones_like(dẋ_dy)

    dẏ_dx = -10*x*np.ones_like(dẋ_dz)
    dẏ_dy = -np.ones_like(dẏ_dx)
    dẏ_dz = np.zeros_like(dẏ_dy)

    dż_dx = μ*s*np.ones_like(dẏ_dz)
    dż_dy = np.zeros_like(dż_dx)
    dż_dz = -μ*np.ones_like(dż_dy)

    j_x = [dẋ_dx, dẋ_dy, dẋ_dz]
    j_y = [dẏ_dx, dẏ_dy, dẏ_dz]
    j_z = [dż_dx, dż_dy, dż_dz]

    return np.vstack([np.hstack(j_x), np.hstack(j_y), np.hstack(j_z)])


def cortex_size(mask, val):
    return int(np.sqrt(mask[mask == val].shape))


def plot_final_state(
    y,
    cortices=None, legend=False,
    title=None, channel=0,
    markers=["ro", "k^", "gX", "bD"],
    ylim=[-1.5, 2.5]
):
    if cortices is None:
        cortices = [[0, y.size]]
    m = iter(markers[:len(cortices)])
    for cortex in cortices:
        plt.plot(range(*cortex), y[-1, channel, cortex[0]:cortex[1]],
                 next(m), label=f"{cortex[0]} - {cortex[1] - 1}")
    if legend:
        plt.legend(loc="best")
    if title:
        plt.title(title)
    plt.ylim(ylim)


def plot_beginning_and_end(y, start, end, p=0.99,
                           legend=False, title=True, channel=0):
    length = y.shape[0]
    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    for i in range(start, end):
        ax1.plot(y[:int(p*length), channel, i], label=i)
        ax1.grid(True)
        ax1.set_xlim([0, int(p*length)])
        ax2.plot(y[int((1 - p)*length):, channel, i], label=i)
        ax2.grid(True)
        ax2.set_xlim([0, int(p*length)])
        plt.ylim([-1.5, 2.25])
    if legend:
        ax1.legend(loc="lower left")
    if title:
        plt.suptitle(f"First and last {100*p}\% of neurons {start} - {end}")


def plot_state_diagram(y, cortices=None, lim=[-1.5, 2.5],
                       markers=["ro", "k^", "gX", "bD"]):
    if cortices is None:
        cortices = [[0, y.size]]
    m = iter(markers[:len(cortices)])
    ytp1 = y[:-1]
    yt = y[1:]
    for cortex in cortices:
        plt.plot(yt[cortex[0]:cortex[1]], ytp1[cortex[0]:cortex[1]], next(m))
    plt.xlim(lim)
    plt.ylim(lim)


def event(t, y, i):
    return y[i]


def time_to_index(t, tmax, N, as_int=True):
    out = (N*t/tmax)
    if as_int:
        return out.astype(int)
    else:
        return out


def firing_time_mask(firing_timeses, tmax, N):
    times = np.linspace(0, tmax, N)
    k_mask = np.zeros((N, len(firing_timeses)))
    t_mask = k_mask.copy()
    tp1_mask = t_mask.copy() + 1
    for i, firing_times in enumerate(firing_timeses):
        for j, firing_time in enumerate(firing_times[:-1]):
            k_mask[times >= firing_time, i] += 1
            t_mask[times >= firing_time, i] = firing_time
            tp1_mask[times >= firing_time, i] = firing_times[j + 1]
    return k_mask.astype(int), t_mask, tp1_mask


def ϕ(sol, t):
    n_areas = len(sol.t_events)
    T = np.vstack(n_areas*[t]).T
    t_events = [sol.t_events[i] for i in range(n_areas)]
    k_mask, t_mask, tp1_mask = firing_time_mask(t_events, t[-1], t.size)
    return 2*np.pi*(T - t_mask)/(tp1_mask - t_mask)


def main():
    parser = argparse.ArgumentParser(description="Simulate a brain")
    parser.add_argument("a", metavar="alpha", type=float, nargs=1,
                        help="The alpha value to use (inter connection strength)")
    parser.add_argument("b", metavar="beta", type=float, nargs=1,
                        help="The beta value to use (intra connection strength)")
    args = parser.parse_args()

    # DATA PREPARATION - specific to the data in question
    data = pd.read_excel("../connectomes/mouse.xlsx", sheet_name=None)
    metadata = pd.read_excel("../connectomes/mouse_meta.xlsx", sheet_name=None)

    m = metadata["Voxel Count_295 Structures"]
    m = m.loc[m["Represented in Linear Model Matrix"] == "Yes"]

    columns = []
    cortices = [[0, 0]]
    for region in m["Major Region"].unique():
        i = [columns.append(acronym) for acronym in
             m.loc[m["Major Region"] == region, "Acronym"].values]
        cortices.append([cortices[-1][-1], cortices[-1][-1] + len(i)])
    cortices.remove([0, 0])

    data = pd.read_excel("../connectomes/mouse.xlsx", sheet_name=None)

    d = data["W_ipsi"]
    p = data["PValue_ipsi"]
    d.columns = columns
    p.columns = columns
    d.index = columns
    p.index = columns

    d = d.values
    p = p.values

    p[np.isnan(p)] = 1

    d[p > 0.01] = 0

    n = np.zeros_like(d)

    for i in [1e-4, 1e-2, 1]:
        n[d >= i] += 1

    G1, G2, events = prep_masks(n, cortices)

    # SPECIFIC PARAMETERS
    b = 3.2                            # Controls spiking frequency
    # Input current --- An array to add noise
    i0 = 4.4*np.ones(n.shape[0])
    x_rev = 2                          # Reverse potential
    λ = 10                             # Sigmoidal function parameter
    θ = -0.25                          # Sigmoidal function parameter
    μ = 0.01                           # Time scale of slow current
    # Governs adaptation (whatever that means)
    s = 4.0
    x_rest = -1.6                      # Resting potential --- INCORRECT IN SANTOS PAPER
    # Intra connection strength ---- VARIED PARAMETER
    α = args.a[0]
    # Number of intra connections from a given neuron
    n1 = np.count_nonzero(G1, axis=1)
    # This is to remove a divide-by-zero; if n1 is 0, then so is G1
    n1[n1 == 0] = 1
    # Inter connection strength ---- VARIED PARAMETER
    β = args.b[0]
    # Number of inter connections from a given neuron
    n2 = np.count_nonzero(G2, axis=1)
    # This is to remove a divide-by-zero; if n2 is 0, then so is G2
    n2[n2 == 0] = 1

    ivs = np.zeros([3, n.shape[0]])    # Initial values [[x], [y], [z]]
    ivs[0] = 3.0*np.random.random(n.shape[0]) - 1.0
    ivs[1] = 0.2*np.random.random(n.shape[0])
    ivs[2] = 0.2*np.random.random(n.shape[0])

    tmax = 4000
    N = 100*tmax
    t = np.linspace(0, tmax, N)

    params = (b, i0, x_rev, λ, θ, μ, s, x_rest, α, n1, β, n2, G1, G2)
    print("Finding solution... ", end=" ")
    sol = solve_ivp(fun=lambda t_in, y_in: hr_dots(y_in, t_in, *params),
                    t_span=(0, tmax), y0=ivs.reshape(ivs.size), events=events,
                    dense_output=True, method="RK45")
    print("Found solution")
    print("Finding phase... ", end=" ")
    phase = ϕ(sol, t)
    print("Found phase")

    print("Writing... ", end=" ")
    with open(f"../../data/{α:0.3f}-{β:0.3f}.pkl", "wb") as f:
        pickle.dump([params, sol, phase], f)

    print("Wrote")


if __name__ == "__main__":
    main()
