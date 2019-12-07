#####################################
# written by Yue Tang
#####################################

from sklearn.multioutput import ClassifierChain
import random
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.utils import validation

class EnsembleClassifierChain(
                              BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    def __init__(
                 self,
                 estimator,
                 number_of_chains=10,
                 threshold=.5,
                 max_features=1.0):
        self.number_of_chains = number_of_chains
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.estimators_ = []
    
    def fit(self, X, y):
        validation.check_X_y(X, y, multi_output=True)
        y = validation.check_array(y, accept_sparse=True)
        
        for i in range(self.number_of_chains):
            # the classifier gets cloned internally in classifer chains, so
            # no need to do that here.
            cc = ClassifierChain(self.estimator)
            
            no_samples = y.shape[0]
            
            # create random subset for each chain individually
            idx = random.sample(range(no_samples),
                                int(no_samples * self.max_features))
                                cc.fit(X[idx, :], y[idx, :])
                                
            self.estimators_.append(cc)

def predict(self, X):
    validation.check_is_fitted(self, 'estimators_')
    
    preds = np.array([cc.predict(X) for cc in self.estimators_])
        preds = np.sum(preds, axis=0)
        W_norm = preds.mean(axis=0)
        out = preds / W_norm
        
        return (out >= self.threshold).astype(int)
