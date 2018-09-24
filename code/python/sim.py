import matplotlib.pyplot as plt
import numpy as np
import pickle

from itertools import cycle
from scipy.integrate import solve_ivp


def Θ(x, x_rev, λ, θ):
    xk, xj = np.meshgrid(x, x)
    return (xj - x_rev)/(1 + np.exp(-λ*(xk - θ)))


def dΘ_dx(x, λ, θ):
    final = np.ones((x.size, x.size))/(1 + np.exp(-λ*(x - θ)))
    np.fill_diagonal(final, final.diagonal() + x*λ*np.exp(-λ*(x - θ))/(1+np.exp(-λ*(x - θ)))**2)
    return final


def hr_dots(current, _, b, i0, x_rev, λ, θ, μ, s, x_rest, α, n1, β, n2, G1, G2):
    x, y, z = map(lambda k: k.flatten(), np.split(current, 3))
    theta = Θ(x, x_rev, λ, θ)
    dots = np.zeros_like(current).reshape(3, -1)
    dots[0] = y - (x**3) + b*(x**2) + i0 - z - (α/n1)*np.sum(G1*theta, axis=1) - (β/n2)*np.sum(G2*theta, axis=1)
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
        plt.plot(range(*cortex), y[-1, channel, cortex[0]:cortex[1]], next(m), label=f"{cortex[0]} - {cortex[1] - 1}")
    if legend:
        plt.legend(loc="best")
    if title:
        plt.title(title)
    plt.ylim(ylim)


def plot_beginning_and_end(y, start, end, p=0.99, legend=False, title=True, channel=0):
    l = y.shape[0]
    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    for i in range(start, end):
        ax1.plot(y[:int(p*l), channel, i], label=i)
        ax1.grid(True)
        ax1.set_xlim([0, int(p*l)])
        ax2.plot(y[int((1 - p)*l):, channel, i], label=i)
        ax2.grid(True)
        ax2.set_xlim([0, int(p*l)])
        plt.ylim([-1.5, 2.25])
    if legend:
        ax1.legend(loc="lower left")
    if title:
        plt.suptitle(f"First and last {100*p}\% of neurons {start} - {end}")


def plot_state_diagram(y, cortices=None, lim=[-1.5, 2.5], markers=["ro", "k^", "gX", "bD"]):
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
event.direction = 1.0


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
    return 2*np.pi*(k_mask + (T - t_mask)/(tp1_mask - t_mask))


run = False
write = False


cortices = [[0, 18],
            [18, 28],
            [28, 46],
            [46, 65]]


n = np.loadtxt("../connectomes/cat_matrix.dat")/3
cortex_mask = np.zeros_like(n)
cortex_mask[:18, :18] = 1
cortex_mask[18:28, 18:28] = 2
cortex_mask[28:46, 28:46] = 3
cortex_mask[46:, 46:] = 4
G1 = n.copy()
G1[cortex_mask == 0] = 0
G2 = n.copy()
G2[cortex_mask != 0] = 0
events = [lambda t_in, y_in, i=i: event(t_in, y_in, i) for i in range(n.shape[0])]
for e in events:
    e.direction = 1.0


# For validation of the cortex sizes
for i in [1, 2, 3, 4]:
    print(i, cortex_size(cortex_mask, i))


plt.matshow(n)
plt.colorbar()


plt.matshow(cortex_mask)
plt.colorbar()


plt.matshow(G1)
plt.colorbar()


plt.matshow(G2)
plt.colorbar()


b = 3.2                           # Controls spiking frequency
i0 = 4.4*np.ones(n.shape[0])      # Input current ---- It's an array so we can add noise later
x_rev = 2                         # Reverse potential
λ = 10                            # Sigmoidal function parameter
θ = -0.25                         # Sigmoidal function parameter
μ = 0.01                          # Time scale of slow current
s = 4.0                           # Governs adaptation (whatever that means)
x_rest = -1.6                     # Resting potential ------ INCORRECT IN SANTOS PAPER
α = 0.210                         # Intra connection strength ---- VARIED PARAMETER
n1 = np.count_nonzero(G1, axis=1) # Number of intra connections from a given neuron
n1[n1 == 0] = 1                   # This is to remove a divide-by-zero; if n1 is 0, then so is G1
β = 0.040                         # Inter connection strength ---- VARIED PARAMETER
n2 = np.count_nonzero(G2, axis=1) # Number of inter connections from a given neuron
n2[n2 == 0] = 1                   # This is to remove a divide-by-zero; if n2 is 0, then so is G2


ivs = np.zeros([3, n.shape[0]])   # Initial values [[x], [y], [z]]
ivs[0] = 3.0*np.random.random(n.shape[0]) - 1.0
ivs[1] = 0.2*np.random.random(n.shape[0])
ivs[2] = 0.2*np.random.random(n.shape[0])


plt.plot(ivs[0], label=r"$x_{0}$")
plt.plot(ivs[1], label=r"$y_{0}$")
plt.plot(ivs[2], label=r"$z_{0}$")
plt.legend()


plt.hist(ivs[0], label=r"$x_{0}$")
plt.hist(ivs[1], label=r"$y_{0}$")
plt.hist(ivs[2], label=r"$z_{0}$")
plt.legend()


tmax = 4000
N = 100*tmax
t = np.linspace(0, tmax, N)


params = (b, i0, x_rev, λ, θ, μ, s, x_rest, α, n1, β, n2, G1, G2)


if run:
    sol_bc = solve_ivp(fun=lambda t_in, y_in: hr_dots(y_in, t_in, *params),
                       t_span=(0, tmax), y0=ivs.reshape(ivs.size), events=events,
                       dense_output=True, method="RK45")
    phase_bc = ϕ(sol_bc, t)
    if write:
        with open("../../data/sol_bc.pkl", "wb") as f:
            pickle.dump(sol_bc, f)
        with open("../../data/phase_bc.pkl", "wb") as f:
            pickle.dump(phase_bc, f)
else:
    with open("../../data/sol_bc.pkl", "rb") as f:
        sol_bc = pickle.load(f)
    with open("../../data/phase_bc.pkl", "rb") as f:
        phase_bc = pickle.load(f)


vals_sync = sol_sync.sol(t).T
vals_sync = vals_sync.reshape(-1, 3, 65)
vals_sync.shape


plot_final_state(y=vals_sync, cortices=cortices, legend=True, title=r"$\alpha = {}$, $\beta = {}$, $t = {}$".format(α, β, tmax))


plot_state_diagram(vals_sync[-1, 0, :], cortices=cortices)


p = 0.03


plot_beginning_and_end(vals_sync, 0, 65, p=p)


plot_beginning_and_end(vals_sync, 0, 18, p=p)


plot_beginning_and_end(vals_sync, 18, 28, p=p)


plot_beginning_and_end(vals_sync, 28, 46, p=p)


plot_beginning_and_end(vals_sync, 46, 65, p=p)


plt.plot(phase_sync)


α = 0.001
β = 0.001


params = (b, i0, x_rev, λ, θ, μ, s, x_rest, α, n1, β, n2, G1, G2)


get_ipython().run_cell_magic('time', '', 'if run:\n    sol_async = solve_ivp(fun=lambda t_in, y_in: hr_dots(y_in, t_in, *params),\n                          t_span=(0, tmax), y0=ivs.reshape(ivs.size), events=events,\n                          dense_output=True, method="RK45")\n    phase_async = ϕ(sol_async, t)\n    if write:\n        with open("../../data/sol_async.pkl", "wb") as f:\n            pickle.dump(sol_async, f)\n        with open("../../data/phase_async.pkl", "wb") as f:\n            pickle.dump(phase_async, f)\nelse:\n    with open("../../data/sol_async.pkl", "rb") as f:\n        sol_async = pickle.load(f)\n    with open("../../data/phase_async.pkl", "rb") as f:\n        phase_async = pickle.load(f)')


vals_async = sol_async.sol(t).T
vals_async = vals_async.reshape(-1, 3, 65)
vals_async.shape


plot_final_state(y=vals_async, cortices=cortices, legend=True, title=r"$\alpha = {}$, $\beta = {}$, $t = {}$".format(α, β, tmax))


plot_state_diagram(vals_async[-1, 0, :], cortices=cortices)


plot_beginning_and_end(vals_async, 0, 65, legend=False, p=p)


plot_beginning_and_end(vals_async, 0, 18, p=p)


plot_beginning_and_end(vals_async, 18, 28, p=p)


plot_beginning_and_end(vals_async, 28, 46, p=p)


plot_beginning_and_end(vals_async, 46, 65, p=p)


α = 0.7
β = 0.08


params = (b, i0, x_rev, λ, θ, μ, s, x_rest, α, n1, β, n2, G1, G2)


get_ipython().run_cell_magic('time', '', 'if run:\n    sol_sc = solve_ivp(fun=lambda t_in, y_in: hr_dots(y_in, t_in, *params),\n                       t_span=(0, tmax), y0=ivs.reshape(ivs.size), events=events,\n                       dense_output=True, method="RK45")\n    phase_sc = ϕ(sol_sc, t)\n    if write:\n        with open("../../data/sol_sc.pkl", "wb") as f:\n            pickle.dump(sol_sc, f)\n        with open("../../data/phase_sc.pkl", "wb") as f:\n            pickle.dump(phase_sc, f)\nelse:\n    with open("../../data/sol_sc.pkl", "rb") as f:\n        sol_sc = pickle.load(f)\n    with open("../../data/phase_sc.pkl", "rb") as f:\n        phase_sc = pickle.load(f)')


vals_sc = sol_sc.sol(t).T
vals_sc = vals_sc.reshape(-1, 3, 65)
vals_sc.shape


plot_final_state(y=vals_sc, cortices=cortices, legend=True, title=r"$\alpha = {}$, $\beta = {}$, $t = {}$".format(α, β, tmax))


plot_state_diagram(vals_sc[-1, 0, :], cortices=cortices)


plot_beginning_and_end(vals_sc, 0, 65, legend=False, p=p)


plot_beginning_and_end(vals_sc, 0, 18, p=p)


vals_sc.shape


plot_beginning_and_end(vals_sc, 18, 28, p=p)


plot_beginning_and_end(vals_sc, 28, 46, p=p)


plot_beginning_and_end(vals_sc, 46, 65, p=p)


α = 1.5
β = 0.1


params = (b, i0, x_rev, λ, θ, μ, s, x_rest, α, n1, β, n2, G1, G2)


get_ipython().run_cell_magic('time', '', 'if run:\n    sol_bc = solve_ivp(fun=lambda t_in, y_in: hr_dots(y_in, t_in, *params),\n                       t_span=(0, tmax), y0=ivs.reshape(ivs.size), events=events,\n                       dense_output=True, method="RK45")\n    phase_bc = ϕ(sol_bc, t)\n    if write:\n        with open("../../data/sol_bc.pkl", "wb") as f:\n            pickle.dump(sol_bc, f)\n        with open("../../data/phase_bc.pkl", "wb") as f:\n            pickle.dump(phase_bc, f)\nelse:\n    with open("../../data/sol_bc.pkl", "rb") as f:\n        sol_bc = pickle.load(f)\n    with open("../../data/phase_bc.pkl", "rb") as f:\n        phase_bc = pickle.load(f)')


vals_bc = sol_bc.sol(t).T
vals_bc = vals_bc.reshape(-1, 3, 65)
vals_bc.shape


plot_final_state(y=vals_bc, cortices=cortices, legend=True, title=r"$\alpha = {}$, $\beta = {}$, $t = {}$".format(α, β, tmax))


plot_state_diagram(vals_bc[-1, 0, :], cortices=cortices)


plot_beginning_and_end(vals_bc, 0, 65, legend=False, p=p)


plot_beginning_and_end(vals_bc, 0, 18, p=p)


plot_beginning_and_end(vals_bc, 18, 28, p=p)


plot_beginning_and_end(vals_bc, 28, 46, p=p)


plot_beginning_and_end(vals_bc, 46, 65, p=p)


ϵ = 0.3


def rp(state, ϵ=ϵ):
    ϕi, ϕj = np.meshgrid(state, state)
    return np.heaviside(ϵ - np.abs(ϕi - ϕj), 0)


def non_central_sparseness(state):
    xx, yy = np.meshgrid(range(state.size), range(state.size))
    return np.sum(np.abs(xx - yy)*rp(state))


for phase in [phase_sync[-1], phase_async[-1], phase_sc[-1], phase_bc[-1]]:
    print(np.sum(rp(phase))/phase.size, non_central_sparseness(phase))


plt.matshow(rp(phase_sync[-1]))


plt.matshow(rp(phase_async[-1]))


plt.matshow(rp(phase_sc[-1]))


plt.matshow(rp(phase_bc[-1]))


def σ(sol):
    diff = [np.diff(ts) for ts in sol.t_events]
    return np.array([np.var(d) for d in diff])


plt.plot(σ(sol_sc))


plt.plot(σ(sol_bc))


plt.plot(σ(sol_sync))


plt.plot(σ(sol_async))
