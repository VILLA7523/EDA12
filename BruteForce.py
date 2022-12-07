"""
KNN - BRUTE FORCE
"""
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

EuclideanDistance = (lambda a, b: math.sqrt(sum(abs(a.get(axis, 0.) - b.get(axis, 0.))**2 for axis in range(len(a)))))

class KNN_BF:
    def __init__(self, data_train, class_train, k):
        self.data_train = data_train
        self.class_train = class_train
        self.k = k
        self.dim = len(self.data_train[0])

    def set_data_test(self, _data_test):
        self.data_test = _data_test

    def set_class_test(self, _data_class):
        self.class_test = _data_class

    def euclidean_distance(self, p1, p2, dim):
        distance = 0.0
        for i in range(0,3):
            distance += (p1[i] - p2[i]) ** 2
        return sqrt(distance)
    def query_neighbors(self, p_query):
#Return a list of k neighbors as(distance, index_neighbor)
        points_dist = []
# 1. Compute all the distances between the query point and reference points.
        idx = 0
        for p_compare in self.data_train:
            _dist = self.euclidean_distance(p_query, p_compare,3)
            p_d = (_dist, idx , p_compare)
            points_dist.append(p_d)
            idx += 1
        points_dist.sort()
        return points_dist[0:self.k]

    def predict(self, ):
        """
        Return a list with classification of data test
        """
        classes = []

# 4. Classification vote by k nearest objects
        for p in self.data_test:
            list_neigh = self.query_neighbors(p)
            votes = []
            for dist, idx in list_neigh:
                votes.append(self.class_train[idx])

            prediction = max(votes, key=votes.count)
            classes.append(prediction)
        return classes