from sklearn import datasets
from sklearn.preprocessing import Imputer
import numpy as np

def isNan(string):
    if string is not '?':
        return int(string)
    else:
        return np.nan     

vowel_data = np.loadtxt('vowel/vowel.train', delimiter=',', skiprows=1)
def loadVowel():
    X = vowel_data[:, -10:]
    y = vowel_data[:, 1].astype(int)
    return (X, y)

cnv = {6: lambda s: isNan(s)}
bc_data = np.loadtxt('wisconsin/breast-cancer-wisconsin.data', delimiter=',', converters=cnv)
bc_data_malignant = bc_data[bc_data[:,10] == 4, :]
bc_data_benign = bc_data[bc_data[:,10] == 2, :]

imp = Imputer(missing_values=np.nan, strategy='median', axis=0)

bc_data = np.concatenate((imp.fit_transform(bc_data_malignant), imp.fit_transform(bc_data_benign))).astype(int)
def loadBC():
    X = bc_data[:, 1:10]
    y = bc_data[:, 10]
    return (X, y)
