from joblib import Parallel, delayed
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sknn.mlp import Classifier, Layer

import data_util as util
import matplotlib.pyplot as plt
import numpy as np

bc_data_train, bc_data_test, bc_target_train, bc_target_test = util.load_breast_cancer()
v_data_train, v_data_test, v_target_train, v_target_test = util.load_vowel()

PORTIONS = np.arange(.1, 1.1, .1)
ITERATIONS = np.arange(250, 20250, 250)
BC_ITER = 7750
V_ITER = 14750
ann_n_iter()
ann_train_size()

def ann_n_iter():
    print "n_iter"
    print "---bc---"
    Parallel(n_jobs=-1)(
        delayed(ann_n_iter)(
            bc_data_train,
            bc_data_test,
            bc_target_train,
            bc_target_test,
            BC_ITER,
            n_iter) for n_iter in ITERATIONS)
    print "---v---"
    Parallel(n_jobs=-1)(
        delayed(ann_n_iter)(
            v_data_train,
            v_data_test,
            v_target_train,
            v_target_test,
            V_ITER,
            n_iter) for n_iter in ITERATIONS)


def _ann_n_iter(data, data_test, target, target_test, n_iter):
    nn = Classifier(
        layers=[
            Layer("Sigmoid", units=100),
            Layer("Softmax")],
        n_iter=n_iter)
    train_score = np.mean(cross_validation.cross_val_score(nn, X_train, y_train, cv=10))
    nn.fit(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    print train_size, train_score, test_score


def ann_train_size():
    print "train_size"
    print "---bc---"
    Parallel(n_jobs=-1)(
        delayed(ann_train_size)(
            bc_data_train, bc_data_test,
            bc_target,
            BC_ITER,
            test_size) for test_size in PORTIONS)
    print "---v---"
    Parallel(n_jobs=-1)(
        delayed(ann_train_size)(
            v_data,
            v_target,
            V_ITER,
            test_size) for test_size in PORTIONS)


def _ann_train_size(data, data_test, target, target_test, n_iter, train_size):
    nn = Classifier(
        layers=[
            Layer("Sigmoid", units=100),
            Layer("Softmax")],
        n_iter=n_iter)
    if train_size < 1:
        X_train, _, y_train, _ = cross_validation.train_test_split(
            data, target, train_size=test_size, stratify=y_train)
    else:
        X_train, y_train = data, target
    train_score = np.mean(cross_validation.cross_val_score(nn, X_train, y_train, cv=10))
    nn.fit(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    print train_size, train_score, test_score
