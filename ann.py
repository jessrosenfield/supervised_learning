from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sknn.mlp import Classifier, Layer

import data_util as util
import matplotlib.pyplot as plt
import numpy as np

def bc_ann_crossval_numitr():
    bc_data, bc_target = util.load_breast_cancer()
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
    print classification_report(params, error)
    fig, ax = plt.subplots()
    ax.scatter(params, errors)
    ax.plot([0, max(params)], [0, max(error)], 'k--', lw=4)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    plt.savefig('/tmp/supervised-learning/bc-ann-crossval-results.png')

def v_ann_crossval_numitr():
    v_data, v_target = util.load_vowel()
    v_nn = Classifier(
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
    print classification_report(params, error)
    fig, ax = plt.subplots()
    ax.scatter(params, errors)
    ax.plot([0, max(params)], [0, max(error)], 'k--', lw=4)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    plt.savefig('/tmp/supervised-learning/v-ann-crossval-results.png')
