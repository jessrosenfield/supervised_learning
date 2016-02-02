from sklearn import datasets
from sklearn.preprocessing import Imputer
import numpy as np

_breast_cancer_path = 'datasets/wisconsin/breast-cancer-wisconsin.data'
_vowel_path = 'datasets/vowel/vowel.train'
_vowel_test_path = 'datasets/vowel/vowel.test'

def _is_nan(string):
    if string is not '?':
        return int(string)
    else:
        return np.nan 

def load_breast_cancer():
    """Load and return the breast cancer wisconsin dataset (classification).
    The breast cancer dataset is a classic and very easy binary classification
    dataset.
    
    Returns
    -------
    (X, y) Tuple
        A tuple of data and target
    
    The copy of UCI ML Breast Cancer Wisconsin (Original) dataset is
    downloaded from:
    http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
    """
    cnv = {6: lambda s: _is_nan(s)}
    bc_data = np.loadtxt(_breast_cancer_path, delimiter=',', converters=cnv)
    bc_data_malignant = bc_data[bc_data[:,10] == 4, :]
    bc_data_benign = bc_data[bc_data[:,10] == 2, :]
    
    imp = Imputer(missing_values=np.nan, strategy='median', axis=0)
    
    bc_data = np.concatenate((imp.fit_transform(bc_data_malignant), imp.fit_transform(bc_data_benign))).astype(int)
    X = bc_data[:, 1:10]
    y = bc_data[:, 10]
    return (X, y)
    
def load_vowel():
    """Load and return the vowel training dataset.
    
    Returns
    -------
    (X, y) Tuple
        A tuple of data and target
    
    The copy of the vowel dataset is downloaded from:
    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    """
    vowel_data = np.loadtxt(_vowel_path, delimiter=',', skiprows=1)
    X = vowel_data[:, -10:]
    y = vowel_data[:, 1].astype(int)
    return (X, y)

def load_vowel_test():
    """Load and return the vowel testing dataset.
    
    Returns
    -------
    (X, y) Tuple
        A tuple of data and target
    
    The copy of the vowel dataset is downloaded from:
    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    """
    vowel_data = np.loadtxt(_vowel_test_path, delimiter=',', skiprows=1)
    X = vowel_data[:, -10:]
    y = vowel_data[:, 1].astype(int)
    return (X, y)

