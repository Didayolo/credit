'''
CREDIT predictive model
This model contains 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np
from os.path import isfile
from sklearn.ensemble import AdaBoostClassifier  

class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=10
        self.num_labels=1
        self.is_trained=False
        self.clf=AdaBoostClassifier()


    def fit(self, X, y):
        '''
        This function trains the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For this binary classification, labels could be either numbers 0 (loan granted) or 1 (load not granted)
        or one-hot vector, it means that it equals to (1,0) if the class is 0 and it's equal to (0,1) if the class is 1
        '''
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]").format(self.num_train_samples, self.num_feat)
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]").format(num_train_samples, self.num_labels)
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.clf.fit(X, y)
        self.is_trained=True



    def predict(self, X):
        '''
        This function provides predictions of labels on (test) data.
        Binary classification with the ROC curve metric problems (as CREDIT problem) often expect predictions
        in the form of a discriminant value.
        '''
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT: dim(X)= [{:d}, {:d}]").format(num_test_samples, num_feat)
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]").format(num_test_samples, self.num_labels)
        y=self.clf.predict(X)
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self