from __future__ import division
from __future__ import print_function

import time
import os

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.gae.optimizer import OptimizerAE, OptimizerVAE
from gae.gae.input_data import load_data, CyberDataset
from gae.gae.model import GCNModelAE, GCNModelVAE
from gae.gae.preprocessing import (
    preprocess_graph,
    construct_feed_dict,
    sparse_to_tuple,
    mask_test_edges,
)


class GAETrainer(object):
    def __init__(self, adj, features, cyber_dataset=None, **params):
        self.adj = adj
        self.features = features
        if cyber_dataset:
            self.cyber_dataset = cyber_dataset

        self.params = {
            "lr": 0.01,
            "epochs": 200,
            "hidden1": 32,
            "hidden2": 16,
            "weight_decay": 0,
            "dropout": 0,
            "model_str": "gcn_ae",
            "hidden1_dim": 32,
            "hidden2_dim": 16,
        }
        self.params.update(params)
        self._setup()

    def _setup(self):
        self.adj_orig = self.adj
        self.adj_orig = self.adj_orig - sp.dia_matrix(
            (self.adj_orig.diagonal()[np.newaxis, :], [0]),
            shape=self.adj_orig.shape,
        )
        self.adj_orig.eliminate_zeros()

        (
            self.adj_train,
            self.train_edges,
            self.val_edges,
            self.val_edges_false,
            self.test_edges,
            self.test_edges_false,
        ) = mask_test_edges(self.adj)
        self.adj = self.adj_train

        # Some preprocessing
        self.adj_norm = preprocess_graph(self.adj)

        # Define placeholders
        self.placeholders = {
            "features": tf.sparse_placeholder(tf.float32),
            "adj": tf.sparse_placeholder(tf.float32),
            "adj_orig": tf.sparse_placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0.0, shape=()),
        }

        num_nodes = self.adj.shape[0]

        self.features = sparse_to_tuple(self.features.tocoo())
        num_features = self.features[2][1]
        features_nonzero = self.features[1].shape[0]

        # Create model
        self.model = None
        if self.params["model_str"] == "gcn_ae":
            self.model = GCNModelAE(
                self.placeholders,
                num_features,
                features_nonzero,
                self.params["hidden1_dim"],
                self.params["hidden2_dim"],
            )
        elif self.params["model_str"] == "gcn_vae":
            self.model = GCNModelVAE(
                self.placeholders, num_features, num_nodes, features_nonzero
            )

        pos_weight = (
            float(self.adj.shape[0] * self.adj.shape[0] - self.adj.sum())
            / self.adj.sum()
        )
        norm = (
            self.adj.shape[0]
            * self.adj.shape[0]
            / float(
                (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2
            )
        )

        # Optimizer
        with tf.name_scope("optimizer"):
            if self.params["model_str"] == "gcn_ae":
                self.opt = OptimizerAE(
                    preds=self.model.reconstructions,
                    labels=tf.reshape(
                        tf.sparse_tensor_to_dense(
                            self.placeholders["adj_orig"],
                            validate_indices=False,
                        ),
                        [-1],
                    ),
                    pos_weight=pos_weight,
                    norm=norm,
                )
            elif self.params["model_str"] == "gcn_vae":
                self.opt = OptimizerVAE(
                    preds=self.model.reconstructions,
                    labels=tf.reshape(
                        tf.sparse_tensor_to_dense(
                            self.placeholders["adj_orig"],
                            validate_indices=False,
                        ),
                        [-1],
                    ),
                    model=self.model,
                    num_nodes=num_nodes,
                    pos_weight=pos_weight,
                    norm=norm,
                )

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_roc_score(self, edges_pos, edges_neg, emb=None):
        if emb is None:
            self.feed_dict.update({self.placeholders["dropout"]: 0})
            emb = self.sess.run(self.model.z_mean, feed_dict=self.feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(self.adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(self.adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def train(self, verbose=False):
        adj_label = self.adj_train + sp.eye(self.adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        val_roc_score = []
        val_ap_score = []

        # Train model
        for epoch in range(self.params["epochs"]):

            t = time.time()
            # Construct feed dictionary
            self.feed_dict = construct_feed_dict(
                self.adj_norm, adj_label, self.features, self.placeholders
            )
            self.feed_dict.update(
                {self.placeholders["dropout"]: self.params["dropout"]}
            )
            # Run single weight update
            outs = self.sess.run(
                [self.opt.opt_op, self.opt.cost, self.opt.accuracy],
                feed_dict=self.feed_dict,
            )

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

            roc_curr, ap_curr = self.get_roc_score(
                self.val_edges, self.val_edges_false
            )
            val_roc_score.append(roc_curr)
            val_ap_score.append(ap_curr)

            if verbose:
                print(
                    "Epoch:",
                    "%04d" % (epoch + 1),
                    "train_loss=",
                    "{:.5f}".format(avg_cost),
                    "train_acc=",
                    "{:.5f}".format(avg_accuracy),
                    "val_roc=",
                    "{:.5f}".format(val_roc_score[-1]),
                    "val_ap=",
                    "{:.5f}".format(ap_curr),
                    "time=",
                    "{:.5f}".format(time.time() - t),
                )

        print("Optimization Finished!")
        return val_roc_score, val_ap_score

    def eval(self):
        roc_score, ap_score = self.get_roc_score(
            self.test_edges, self.test_edges_false
        )
        print("Test ROC score: " + str(roc_score))
        print("Test AP score: " + str(ap_score))
        return roc_score, ap_score
