from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import data_util as util
import matplotlib.pyplot as plt
import numpy as np

print "---bc---"
bc_data, bc_target = util.load_breast_cancer()
bc_knn = KNeighborsClassifier(weights='distance', n_jobs=-1)
n_neighbors = np.arange(1, 11)
clf = GridSearchCV(bc_knn, dict(n_neighbors=n_neighbors), n_jobs=-1, cv=10, verbose=2)
clf.fit(bc_data, bc_target)
print "Best parameters: ", clf.best_params_
params = []
error = []
for gs in clf.grid_scores_:
    print gs[0]['n_neighbors'], gs[1]
print "---v---"
v_data, v_target = util.load_vowel()
v_knn = KNeighborsClassifier(weights='distance', n_jobs=-1)
clr = GridSearchCV(v_knn, dict(n_neighbors=n_neighbors), n_jobs=-1, cv=10, verbose=2)
clf.fit(v_data, v_target)
print "Best parameters: ", clf.best_params_
params = []
error = []
for gs in clf.grid_scores_:
    print gs[0]['n_neighbors'], gs[1]
v_data, v_target = util.load_vowel()


