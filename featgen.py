import networkx as nx
import numpy as np
import random
import abc
import pdb

class FeatureGen(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

class ConstFeatureGen(FeatureGen):
    def __init__(self, val, max_degree=20):
        self.val = val
        self.max_degree = max_degree
        self.one_hots = np.eye(max_degree, dtype=np.float32)

    def gen_node_features(self, G):
        feat_dict = {}
        if self.val is None:
            for k, v in G.degree():
                if v < self.max_degree:
                    feat_dict[k] = {'feat': self.one_hots[v]}
                else:
                    feat_dict[k] = {'feat': self.one_hots[self.max_degree - 1]}
        else:
            feat_dict = {i:{'feat': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

class GaussianFeatureGen(FeatureGen):
    def __init__(self, mu, sigma):
        self.mu = mu
        if sigma.ndim < 2:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        feat_dict = {
                i: {"feat": feat[i]} for i in range(feat.shape[0])
            }
        nx.set_node_attributes(G, feat_dict)

class GridFeatureGen(FeatureGen):
    def __init__(self, mu, sigma, com_choices):
        self.mu = mu
        self.sigma = sigma
        self.com_choices = com_choices

    def gen_node_features(self, G):
        community_dict = {
            n: self.com_choices[0] if G.degree(n) < 4 else self.com_choices[1]
            for n in G.nodes()
        }

        s = np.random.normal(self.mu, self.sigma, G.number_of_nodes())

        feat_dict = {
            n: {"feat": np.asarray([community_dict[n], s[i]])}
            for i, n in enumerate(G.nodes())
        }

        nx.set_node_attributes(G, feat_dict)
        return community_dict
