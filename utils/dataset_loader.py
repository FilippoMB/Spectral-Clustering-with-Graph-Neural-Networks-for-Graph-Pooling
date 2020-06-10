import os
import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from spektral.utils import nx_to_numpy
from utils.misc import node_feat_norm


def read_graphs_txt(ds_name):
    """
    Read benchmark datasets for graph classification.
    Return a list of networkx graphs
    """
    pre = ""

    with open("data/classification/" + pre + ds_name + "/" + ds_name + "_graph_indicator.txt", "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]

    # Nodes
    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0

    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    graph_list = []
    vertex_list = []
    for i in node_indices:
        g = nx.Graph(directed=False)
        vertex_list_g = []
        for j in range(i[1] - i[0] + 1):
            vertex_list_g.append(g.add_node(j))

        graph_list.append(g)
        vertex_list.append(vertex_list_g)

    # Edges
    with open("data/classification/" + pre + ds_name + "/" + ds_name + "_A.txt", "r") as f:
        edges = [i.split(',') for i in list(f)]

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]

    edge_indicator = []
    edge_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        edge_indicator.append(g_id)
        g = graph_list[g_id]
        off = offset[g_id]

        # Avoid multigraph
        edge_list.append(g.add_edge(e[0] - off, e[1] - off))

    # Node labels
    if os.path.exists("data/classification/" + pre + ds_name + "/" + ds_name + "_node_labels.txt"):
        with open("data/classification/" + pre + ds_name + "/" + ds_name + "_node_labels.txt", "r") as f:
            node_labels = [int(i) for i in list(f)]

        i = 0
        for g in graph_list:
            for n in g.nodes():
                g.node[n]['label'] = node_labels[i]
                i += 1

    # Node Attributes
    if os.path.exists("data/classification/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt"):
        with open("data/classification/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt", "r") as f:
            node_attributes = [map(float, i.split(',')) for i in list(f)]

        i = 0
        for g in graph_list:
            for n in g.nodes():
                g.node[n]['attributes'] = list(node_attributes[i])
                i += 1

    # Classes
    with open("data/classification/" + pre + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]

    return graph_list, classes


def get_graph_kernel_dataset(dataset_ID, feat_norm='zscore'):
    print('Loading data')
    nx_graphs, y = read_graphs_txt(dataset_ID)

    # Preprocessing
    y = np.array(y)[..., None]
    y = OneHotEncoder(sparse=False, categories='auto').fit_transform(y)

    # Get node attributes
    try:
        A, X_attr, _ = nx_to_numpy(nx_graphs, nf_keys=['attributes'], auto_pad=False)
        X_attr = node_feat_norm(X_attr, feat_norm)
    except KeyError:
        print('Featureless nodes')
        A, X_attr, _ = nx_to_numpy(nx_graphs, auto_pad=False)  # na will be None

    # Get clustering coefficients (always zscore norm)
    clustering_coefficients = [np.array(list(nx.clustering(g).values()))[..., None] for g in nx_graphs]
    clustering_coefficients = node_feat_norm(clustering_coefficients, 'zscore')

    # Get node degrees
    node_degrees = np.array([np.sum(_, axis=-1, keepdims=True) for _ in A])
    node_degrees = node_feat_norm(node_degrees, feat_norm)

    # Get node labels (always ohe norm)
    try:
        _, X_labs, _ = nx_to_numpy(nx_graphs, nf_keys=['label'], auto_pad=False)
        X_labs = node_feat_norm(X_labs, 'ohe')
    except KeyError:
        print('Label-less nodes')
        X_labs = None

    # Concatenate features
    Xs = [node_degrees, clustering_coefficients]
    if X_attr is not None:
        Xs.append(X_attr)
    if X_labs is not None:
        Xs.append(X_labs)
    X = [np.concatenate(x_, axis=-1) for x_ in zip(*Xs)]
    X = np.array(X)

    return A, X, y

