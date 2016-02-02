from sklearn import cross_validation
from sknn.mlp import Classifier, Layer
import LoadDatasets

bcData, bcTarget = LoadDatasets.loadBC()
bc_nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Softmax")])
scores = cross_validation.cross_val_score(bc_nn, bcData, bcTarget, cv=10)
print scores
