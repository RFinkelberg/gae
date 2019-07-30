import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from os.path import join
from functools import partial
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset)
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1
        )
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


class CyberDataset(object):
    def __init__(self, data_dir):
        self._data_dir = partial(join, data_dir)
        self._edge_list = None
        self._adj = None
        self._features = None

    @property
    def adj(self):
        return self._adj

    @property
    def features(self):
        return self._features

    @property
    def edge_list(self):
        return self._edge_list

    def _adj_from_edgelist(self, edge_list, encoded=False):
        if not encoded:
            edge_list = (
                LabelEncoder().fit_transform(edge_list.reshape(-1)).reshape(-1, 2)
            )
        n = edge_list.max()
        adj = np.zeros((n + 1, n + 1))
        for e in edge_list:
            adj[e[0], e[1]] = 1
            adj[e[1], e[0]] = 1
        adj = sp.csr_matrix(adj, dtype=np.int64)
        return adj

    def load_data(self, dataset):
        if dataset == "email":
            data_path = self._data_dir("email-Eu-core-temporal.txt")
            labels_path = self._data_dir("email-Eu-core-department-labels.txt")
            adj, features = self._load_email(data_path, labels_path)
        elif dataset == "alamos_flows":
            path = self._data_dir("flows.txt")
            adj, features = self._load_alamos_flows(path)
        elif dataset == "UNSW":
            path = self._data_dir("UNSW-NB15_1.csv")
            adj, features = self._load_unsw(path)
        elif dataset == "gnutella":
            path = self._data_dir("p2p-Gnutella08.txt")
            adj, features = self._load_gnutella(path)
        elif dataset == "bitcoin":
            path = self._data_dir("soc-sign-bitcoinalpha.csv")
            adj, features = self._load_bitcoin(path)
        else:
            raise NotImplementedError()

        self._adj = adj
        self._features = features
        return adj, features

    def _load_email(self, data_path, labels_path):
        email_data = np.genfromtxt(data_path, delimiter=" ")
        self._edge_list = email_data
        dept_labels = np.genfromtxt(labels_path, delimiter=" ")
        G = nx.from_edgelist(email_data[:, :2])
        adj = nx.adjacency_matrix(G)

        # define features based on department membership
        features = np.zeros(adj.shape[0])
        for i, node in enumerate(G.nodes()):
            features[i] = dept_labels[int(node), 1]
        features = sp.spdiags(features, 0, adj.shape[0], adj.shape[0])
        return adj, features

    def _load_alamos_flows(self, path):
        edge_list = np.genfromtxt(
            path,
            delimiter=",",
            max_rows=1000000,
            usecols=(2, 4),
            dtype=str,
        )
        self._edge_list = edge_list
        adj = self._adj_from_edgelist(edge_list)
        features = sp.identity(adj.shape[0])
        # TODO experiment with putting in features
        return adj, features

    def _load_unsw(self, path):
        edge_list = np.genfromtxt(
            path, delimiter=",", usecols=(0, 2), dtype=str, encoding='UTF-8'
        )
        self._edge_list = edge_list
        adj = self._adj_from_edgelist(edge_list)
        features = sp.identity(adj.shape[0])
        # TODO experiment with putting in features
        return adj, features

    def _load_gnutella(self, path):
        edge_list = np.genfromtxt(path, delimiter="\t")
        self._edge_list = edge_list
        adj = self._adj_from_edgelist(edge_list)
        features = sp.identity(adj.shape[0])
        return adj, features

    def _load_amazon(self, path): 
        """ NOTE currently unused in experiments. The dataset was too
        large to train, and subsampling was unreasonable
        """
        edge_list = np.genfromtxt(path, delimiter="\t", max_rows=1000)
        self._edge_list = edge_list
        adj = self._adj_from_edgelist(edge_list)
        features = sp.identity(adj.shape[0])
        return adj, features

    def _load_brightkite(self, path):
        edge_list = np.genfromtxt(path, delimiter="\t", max_rows=10000)
        self._edge_list = edge_list
        adj = self._adj_from_edgelist(edge_list)
        features = sp.identity(adj.shape[0])
        return adj, features
    
    def _load_bitcoin(self, path):
        edge_list = np.genfromtxt(path, delimiter=",", usecols=(0, 1, 2))
        self._edge_list = edge_list
        adj = self._adj_from_edgelist(edge_list[:, :2])
        features = sp.identity(adj.shape[0])
        return adj, features
    
    def _load_cic_ids(self, path):
        edge_list = np.genfromtxt(path, delimiter=",", usecols=(1, 3), max_rows=10000)
        self._edge_list = edge_list
        adj = self._adj_from_edgelist(edge_list)
        features = sp.identity(adj.shape[0])
        return adj, features