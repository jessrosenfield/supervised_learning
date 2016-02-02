from sklearn import cross_validation
from sknn.mlp import Classifier, Layer
import data_util as util

bc_data, bc_target = util.load_breast_cancer()
bc_nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Softmax")])
scores = cross_validation.cross_val_score(bc_nn, bc_data, bc_target, cv=10)
print scores

