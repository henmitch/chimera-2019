import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

parser = argparse.ArgumentParser(description="Plot a timestep in a network")
parser.add_argument(
    "t", metavar="time", type=int, nargs=1,
    help="The timestep at which to plot"
)
parser.add_argument(
    "-min", metavar="min", type=float, nargs=1, default=-np.pi,
    help="The minimum potential value (defaults to -pi)"
)
parser.add_argument(
    "-max", metavar="max", type=float, nargs=1, default=np.pi,
    help="The maximum potential value (defaults to pi)"
)
parser.add_argument(
    "f", metavar="file", type=str, nargs=1,
    help="The gexf file containing the network"
)

args = parser.parse_args()
t = args.t[0]

G = nx.read_gexf(args.f[0], version="1.2draft")


pos = {i: [G.nodes[i]["viz"]["position"]["x"],
           G.nodes[i]["viz"]["position"]["y"]] for i in G.nodes}


node_size = np.array([G.nodes[i]["viz"]["size"] for i in G.nodes])


keys = [G.nodes[i].keys() for i in G.nodes]


potentials = np.array([G.nodes[i]["potential"][t][0] for i in G.nodes])
nx.draw_networkx(G, pos=pos, node_size=node_size, arrowsize=1, width=0.05,
                 edge_color="#777777", node_color=potentials,
                 with_labels=False, vmin=args.min, vmax=args.max)
plt.xticks([])
plt.yticks([])

plt.savefig(f"animated/{t:04}.png", dpi=100)
