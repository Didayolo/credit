import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
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


# grid search
model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

# classifier

val_score = roc_auc_score(y_val , grid_result.best_estimator_.predict(X_val))
test_score = roc_auc_score(y_test , grid_result.best_estimator_.predict(X_test))
print "AUC score on the validation set = %.5f " % val_score
print "AUC score on the test set = %.5f " % test_score
print "Average score", (val_score+test_score)/float(2)

# plot
pyplot.errorbar(learning_rate, means, yerr=stds)
pyplot.title("XGBoost learning_rate vs ROC AUC score")
pyplot.xlabel('learning_rate')
pyplot.ylabel('ROC AUC score')
pyplot.savefig('learning_rate.png')





