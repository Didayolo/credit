from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier

#load data
X_train = np.loadtxt('./data/credit_train.data', delimiter = ' ')
y_train = np.loadtxt('./data/credit_train.solution')
y_val = np.loadtxt('./data/credit_valid.solution')
y_test = np.loadtxt('./data/credit_test.solution')
X_val = np.loadtxt('./data/credit_valid.data', delimiter = ' ')
X_test = np.loadtxt('./data/credit_test.data', delimiter = ' ')

# upsampling
Xy = np.hstack((X_train,y_train.reshape(-1,1)))
majority = Xy[Xy[:,Xy.shape[1]-1]==0]
minority = Xy[Xy[:,Xy.shape[1]-1]==1]
minority_upsampled = resample(minority,
                                 replace=True,
                                 n_samples=len(Xy[Xy[:,Xy.shape[1]-1]==0]),
                                 random_state=123)
upsampled = np.vstack([majority, minority_upsampled])
np.random.shuffle(upsampled)
X_train = upsampled[:,:-1]
y_train = upsampled[:,-1]

# classifier
clf = MLPClassifier()
clf.fit(X_train, y_train)
val_score = roc_auc_score(y_val , clf.predict(X_val))
test_score = roc_auc_score(y_test , clf.predict(X_test))
print "AUC score on the validation set = %.5f " % val_score
print "AUC score on the test set = %.5f " % test_score
print "Average score", (val_score+test_score)/float(2)
print "Classifier details", clf
