from functools import partial

import cudf
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

import cugraph


class LinkPredictor(object):
    def __init__(self, metric=nx.link_prediction.adamic_adar_index):
        self._metric = metric
        self._threshold = 0

    def train(self, G_train, val_edges, val_edges_false):
        self._metric = partial(self._metric, G_train)
        surface = np.array(
            list(self._metric((tuple(e) for e in val_edges)))
            + list(self._metric((tuple(e) for e in val_edges_false)))
        )

        actual = np.array([1] * len(val_edges) + [0] * len(val_edges_false))

        def _func(threshold):
            pred = surface[:, 2] > threshold
            return roc_auc_score(actual, pred)

        self._threshold = max(np.arange(0.0, 1.0, 0.01), key=_func)
        return self

    def predict(self, edges):
        ret = np.zeros(len(edges))
        for i, (_, _, conf) in enumerate(
            self._metric(tuple(e) for e in edges)
        ):
            if conf > self._threshold:
                ret[i] = 1
        return ret


class CugraphPredictor(object):
    def __init__(self, G):
        self._threshold = 0
        self._G = G

    def train(self, val_edges, val_edges_false):
        validation_set = np.array(
            list(tuple(e) for e in val_edges)
            + list(tuple(e) for e in val_edges_false)
        )
        surface = cugraph.jaccard(
            self._G,
            first=cudf.Series(validation_set[:, 0]).astype("int32"),
            second=cudf.Series(validation_set[:, 1]).astype("int32"),
        )
        actual = np.array([1] * len(val_edges) + [0] * len(val_edges_false))

        def _func(threshold):
            pred = surface.iloc[:, 2] > threshold
            return roc_auc_score(actual, pred)

        self._threshold = max(np.arange(0.0, 1.0, 0.01), key=_func)
        return self

    def predict(self, first, second):
        return (
            cugraph.jaccard(
                self._G, first.astype("int32"), second.astype("int32")
            )
            > self._threshold
        )
