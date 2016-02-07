from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import data_util as util
import matplotlib.pyplot as plt
import numpy as np

K_NEIGHBORS = np.arange(1, 11)
bc_data_train, bc_data_test, bc_target_train, bc_target_test = util.load_breast_cancer()
v_data_train, v_data_test, v_target_train, v_target_test = util.load_vowel()

def knn_neighbors():
    print "knn_k_neighbors"
    print "---bc---"
    Parallel(n_jobs=-1)(
        delayed(_knn_neighbors)(
            bc_data_train,
            bc_data_test,
            bc_target_train,
            bc_target_test,
            n_neighbors) for n_neighbors in K_NEIGHBORS)
    print "---v---"
    Parallel(n_jobs=-1)(
        delayed(_knn_neighbors)(
            v_data_train,
            v_data_test,
            v_target_train,
            v_target_test,
            n_neighbors) for n_neighbors in K_NEIGHBORS)


def _knn_neighbors(data, data_test, target, target_test, n_iter):
    knn = KNeighborsClassifier(weights='distance', n_jobs=-1)
    train_score = np.mean(cross_validation.cross_val_score(knn, data, target, cv=10))
    knn.fit(data, target)
    test_score = knn.score(data_test, target_test)
    print n_iter, train_score, test_score

knn_neighbors()
