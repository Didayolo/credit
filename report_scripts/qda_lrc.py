import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import seaborn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot


#load data
X_train = np.loadtxt('./data/data_matrices/credit_train.data', delimiter = ' ')
y_train = np.loadtxt('./data/data_matrices/credit_train.solution')
y_val = np.loadtxt('./data/data_matrices/credit_valid.solution')
y_test = np.loadtxt('./data/data_matrices/credit_test.solution')
X_val = np.loadtxt('./data/data_matrices/credit_valid.data', delimiter = ' ')
X_test = np.loadtxt('./data/data_matrices/credit_test.data', delimiter = ' ')
print X_train.shape, X_val.shape, X_test.shape

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
clf =  QuadraticDiscriminantAnalysis()

val_scores = []
test_scores = []
x = []
means = []

arr = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 226758]

for n in arr:
	x.append(n)
	clf.fit(X_train[:n,:], y_train[:n])
	val_score = roc_auc_score(y_val , clf.predict(X_val))
	test_score = roc_auc_score(y_test , clf.predict(X_test))
	mean = (val_score+test_score)/float(2)
	val_scores.append(val_score)
	test_scores.append(test_score)
	means.append(mean)
	print n 
	print "AUC score on the validation set = %.5f " % val_score
	print "AUC score on the test set = %.5f " % test_score
	print "Average score", (val_score+test_score)/float(2)

pyplot.plot(x, means, label='Average score')
pyplot.plot(x, val_scores, label='Validation score')
pyplot.plot(x, test_scores, label='Test scores score')

pyplot.legend()
pyplot.xlabel('Number of samples')
pyplot.ylabel('AUC ROC score')
pyplot.savefig('qda_lc.png')


