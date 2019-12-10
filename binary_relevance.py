import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings

#####################################
# 
#####################################

warnings.filterwarnings("ignore")
def binary_relevance(X_train, y_train, X_test):
    classifiers = []
    for i in range(y_train.shape[1]):
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train[:,i])
        classifiers.append(classifier)
    predictions = []
    for classifier in classifiers:
        pred = np.array(classifier.predict(X_test)).T
        predictions.append(pred)
    predictions = np.array(predictions).T
    return predictions
