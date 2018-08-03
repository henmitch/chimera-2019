# filename:      convert_connectome.py
# author:        Henry Mitchell
# creation date: 27 Apr 2018

# description:   Convert a connectome from graphml format to a numpy array
#                and save as an array

import argparse
import numpy as np
import pickle
import re

from graph_tool import load_graph
from graph_tool.spectral import adjacency

parser = argparse.ArgumentParser()
parser.add_argument(
    "connectome", metavar="connectome", type=str,
    help="path to the connectome in question"
)
parser.add_argument(
    "--pickle", default=False, action="store_true",
    help="if the output should be pickled"
)
parser.add_argument(
    "--output_file", required=False,
    help="path to the output file"
)
args = parser.parse_args()

if args.output_file:
    filename = args.output_file
else:
    head = re.match(r".*(?=\.[^\.]+$)", args.connectome).group()
    if args.pickle:
        filename = head + ".pickle"
    else:
        filename = head + ".txt"


print("Loading graph")
g = load_graph(args.connectome)
n = adjacency(
    g,
    weight=g.edge_properties["number_of_fiber_per_fiber_length_mean"]
).todense()

if args.pickle:
    print("Pickling to", filename)
    with open(filename, "wb") as f:
        pickle.dump(n, f)
else:
    print("Writing to", filename)
    np.savetxt(filename, n)

print("Done.")
