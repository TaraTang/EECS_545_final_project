#####################################
# written by Yue Tang
#####################################

import numpy as np
from numpy import genfromtxt
from scipy.sparse import hstack
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

def convert_X(mat):
    mat = csr_matrix(mat)
    return mat

def modified_chain_log(X_train, y_train, X_test, threshold=0.2):
    X1 = X_train
    X2 = X_test
    classifiers = []
    indices = []
    for i in range(y_train.shape[1]):
        classifier = LogisticRegression()#class_weight = 'balanced')
        added_y = []
        if i >= 1:
            for j in range(i):
                corr = np.corrcoef(y_train[:,j],y_train[:,i])[0][1]
                if corr >= threshold or corr <= -threshold:
                    added_y.append(j)
        if len(added_y) > 0:
            classifier.fit(np.hstack([X1,(y_train[:,added_y])]), y_train[:,i])
        else:
            classifier.fit(X1, y_train[:,i])
        classifiers.append(classifier)
        indices.append(added_y)
    #print (i)
    predictions = []
    for i in range(len(indices)):
        X_original = X2.copy()
        classifier = classifiers[i]
        index = indices[i]
        if len(index) >= 1:
            for ii in index:
                X2 = np.hstack([X2,predictions[ii].reshape(X2.shape[0],1)])
            pred = np.array(classifier.predict(X2)).T
        else:
            pred = np.array(classifier.predict(X2)).T
        predictions.append(pred)
        X2 = X_original
    predictions = np.array(predictions).T
    return predictions
