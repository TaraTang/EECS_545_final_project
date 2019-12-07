import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings

#####################################
# written by Yue Tang
#####################################

warnings.filterwarnings("ignore")
def class_chain_log(X_train, y_train, X_test):
    classifiers = []
    for i in range(y_train.shape[1]):
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train[:,i])
        classifiers.append(classifier)
        X_train = np.hstack([X_train, (y_train[:,i]).reshape(X_train.shape[0],1)])
    predictions = []
    for classifier in classifiers:
        pred = np.array(classifier.predict(X_test)).T
        predictions.append(pred)
        X_test = np.hstack([X_test, pred.reshape(X_test.shape[0],1)])
    predictions = np.array(predictions).T
    return predictions
