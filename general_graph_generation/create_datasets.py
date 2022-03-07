import networkx as nx
import numpy as np

from utils import *
from data import *
from create_graphs import *

import argparse
def _parse_args():
    parser = argparse.ArgumentParser(description = 'General graph generation')
    parser.add_argument('--graph_type', '-graph_type', type = str, default = 'grid', help = 'Graph type')
    parser.add_argument('--num_communities', '-num_communities', type = int, default = 2, help = 'Number of communities')
    args = parser.parse_args()
    return args
args = _parse_args()

# Tree
print('Tree')
args.graph_type = 'tree'
graphs = create(args)
print(len(graphs))
print(graphs[0].number_of_nodes())
print(graphs[0].number_of_edges())
print(graphs[0].nodes.data())
A = nx.adjacency_matrix(graphs[0]).todense()
print(A)

# Community
print('Community')
args.graph_type = 'community'
graphs = create(args)
print(len(graphs))
print(graphs[0].number_of_nodes())
print(graphs[0].number_of_edges())

