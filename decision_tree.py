import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

import data_util as util

bc_data, bc_target = util.load_breast_cancer()
v_data, v_target = util.load_vowel()

def bc_dc_gridsearch:
  bc_dc = DecisionTreeClassifier()
  param_grid = {'criterion': ["gini", "entropy"]}
  clf = GridSearchCV(bc_dc, param_grid, n_jobs=-1, cv=10, verbose=2)
  clf.fit(bc_data, bc_target)
  print "Best estimator: ", clf.best_estimator_
  print "Best parameters set found: ", clf.best_params_
  print classification_report(clf.grid_scores_)
  
def v_dc_gridsearch:
  v_dc = DecisionTreeClassifier()
  param_grid = {'criterion': ["gini", "entropy"]}
  clf = GridSearchCV(v_dc, param_grid, n_jobs=-1, cv=10, verbose=2)
  clf.fit(v_data, v_target)
  print "Best estimator: ", clf.best_estimator_
  print "Best parameters set found: ", clf.best_params_
  print classification_report(clf.grid_scores_)
