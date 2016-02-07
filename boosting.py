from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

import data_util as util

bc_data_train, bc_data_test, bc_target_train, bc_target_test = util.load_breast_cancer()
v_data_train, v_data_test, v_target_train, v_target_test = util.load_vowel()

iris = load_iris()


if __name__ == "__main__":
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(bc_data_train, bc_target_train)
    train_score = clf.score(bc_data_train, bc_target_train)
    test_score = clf.score(bc_data_test, bc_target_test)
    "print"