from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sknn.mlp import Classifier, Layer

import data_util as util
import matplotlib.pyplot as plt
import numpy as np

bc_data, bc_target = util.load_breast_cancer()
bc_nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Softmax")])
num_iterations = np.arange(500, 20250, 250)
param_grid = {'n_iter': num_iterations}
clf = GridSearchCV(bc_nn, param_grid, n_jobs=-1, cv=10, verbose=2)
clf.fit(bc_data, bc_target)
print "Best estimatore: ", clf.best_estimator_
print "Best parameters set found: ", clf.best_params_
print classification_report(clf.grid_scores_)

fig, ax = plt.subplots()
ax.scatter(num_iterations, mean_errors)
ax.plot([0, num_iterations.max()], [0, mean_errors.max()], 'k--', lw=4)
ax.set_xlabel('')
ax.set_ylabel('Predicted')
plt.savefig('ann-crossval-results.png')
plt.show()

