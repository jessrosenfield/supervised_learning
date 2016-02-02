from sklearn import datasets
from sklearn.svm import SVC
from sknn.mlp import Classifier, Layer

wdbc = datasets.load_breast_cancer()

svm = SVC()
svm.fit(wdbc.data, wdbc.target)

nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=10)
nn.fit(wdbc.data, wdbc.target)

