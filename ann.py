from joblib import Parallel, delayed
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sknn.mlp import Classifier, Layer

import data_util as util
import matplotlib.pyplot as plt
import numpy as np

bc_data, bc_target = util.load_breast_cancer()
v_data, v_target = util.load_vowel()
portions = np.arange(.1, 1, .1)
BC_ITER = 7750
V_ITER = 14750


def bc_ann_crossval_numitr():
    print "---breast-cancer-wisconsin---"
    bc_nn = Classifier(
        layers=[
            Layer("Sigmoid", units=100),
            Layer("Softmax")])
    num_iterations = np.arange(500, 20250, 250)
    param_grid = {'n_iter': num_iterations}
    clf = GridSearchCV(bc_nn, param_grid, n_jobs=-1, cv=10)
    clf.fit(bc_data, bc_target)
    print "Best parameters: ", clf.best_params_
    params = []
    error = []
    for gs in clf.grid_scores_:
        params.append(gs[0]['n_iter'])
        error.append(gs[1])
        print gs[0]['n_iter'], gs[1]
    fig, ax = plt.subplots()
    ax.scatter(params, errors)
    ax.plot([0, max(params)], [0, max(error)], 'k--', lw=4)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    plt.savefig('/tmp/supervised-learning/bc-ann-crossval-results.png')

def v_ann_crossval_numitr():
    print "---vowel---"
    v_nn = Classifier(
        layers=[
            Layer("Sigmoid", units=100),
            Layer("Softmax")])
    num_iterations = np.arange(500, 20250, 250)
    param_grid = {'n_iter': num_iterations}
    clf = GridSearchCV(v_nn, param_grid, n_jobs=-1, cv=10)
    clf.fit(v_data, v_target)
    print "Best parameters: ", clf.best_params_
    params = []
    error = []
    for gs in clf.grid_scores_:
        params.append(gs[0]['n_iter'])
        error.append(gs[1])
        print gs[0]['n_iter'], gs[1]
    fig, ax = plt.subplots()
    ax.scatter(params, errors)
    ax.plot([0, max(params)], [0, max(error)], 'k--', lw=4)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    plt.savefig('/tmp/supervised-learning/v-ann-crossval-results.png')

def ann_test_size_test():
    print "---bc---"
    Parallel(n_jobs=-1)(delayed(ann_varied_test_size)(bc_data, bc_target, BC_ITER, test_size) for test_size in portions)
    print "---v---"
    Parallel(n_jobs=-1)(delayed(ann_varied_test_size)(v_data, v_target, V_ITER, test_size) for test_size in portions)

def ann_varied_test_size(data, target, n_iter, test_size):
    nn = Classifier(
        layers=[
            Layer("Sigmoid", units=100),
            Layer("Softmax")],
        n_iter=n_iter)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=test_size)
    nn.fit(X_train, y_train)
    score = nn.score(X_test, y_test)
    print test_size, score
