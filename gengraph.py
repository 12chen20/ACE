import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

plt.switch_backend("agg")
import networkx as nx
import numpy as np
import synthetic_structsim
import featgen
import pdb


def perturb(graph_list, p):
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def generate_graph(basis_type="ba",
                   shape="house",
                   nb_shapes=80,
                   width_basis=300,
                   feature_generator=None,
                   m=5,
                   random_edges=0.0):
    if shape == "house":
        list_shapes = [["house"]] * nb_shapes
    elif shape == "cycle":
        list_shapes = [["cycle", 6]] * nb_shapes
    elif shape == "diamond":
        list_shapes = [["diamond"]] * nb_shapes
    elif shape == "grid":
        list_shapes = [["grid"]] * nb_shapes
    else:
        assert False

    G, role_id, _ = synthetic_structsim.build_graph(width_basis,
                                                    basis_type,
                                                    list_shapes,
                                                    rdm_basis_plugins=True,
                                                    start=0,
                                                    m=m)

    if random_edges != 0:
        G = perturb([G], random_edges)[0]
    feature_generator.gen_node_features(G)
    return G, role_id

