import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import seaborn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

# ploting tuning LR
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

means = [0.830133 , 0.831271 , 0.855723 , 0.875144 , 0.881350 , 0.886054]

stds = [0.002662, 0.002579, 0.002840, 0.002786, 0.002799, 0.002483]


pyplot.errorbar(learning_rate, means, yerr=stds)
pyplot.title("XGBoost learning_rate vs ROC AUC score")
pyplot.xlabel('learning_rate')
pyplot.ylabel('ROC AUC score')
pyplot.savefig('learning_rate.png')


# ploting tuning LR + trees

n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
means = [0.830286, 0.830296, 0.830330 , 0.830339 , 0.830555 , 0.831218 , 0.842958 , \
0.848466, 0.850072 , 0.850529 , 0.855643, 0.860593 , 0.864613 , 0.867667 , 0.869497 , \
0.875011 , 0.881197 , 0.886399 , 0.890985 , 0.895406 ]
errors = [0.003222, 0.003219, 0.003229, 0.003208, 0.003244, 0.003661, 0.003512, 0.003077, \
0.003177, 0.003021, 0.003061, 0.002893, 0.002800, 0.002732, 0.002694, 0.002601, 0.002653, \
0.002521, 0.002405, 0.002447] 
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
err = np.array(errors).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    pyplot.errorbar(n_estimators, scores[i], yerr=err[i] , label='learning_rate: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('AUC ROC score')
pyplot.savefig('n_estimators_vs_learning_rate.png')






