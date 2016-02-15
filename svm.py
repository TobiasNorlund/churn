
import numpy as np
from sklearn.svm import *
import sklearn.cross_validation
import data_loader
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectFromModel


# Load train data
(X_train, Y_train) = data_loader.load("Dataset/churn.data.txt")
(X_test, Y_test) = data_loader.load("Dataset/churn.test.txt")

def custom_scorer(ground_truth, predictions):
    ground_truth = ground_truth
    predictions = predictions
    prec = sklearn.metrics.precision_score(ground_truth, predictions)
    rec = sklearn.metrics.recall_score(ground_truth, predictions)
    f1 = sklearn.metrics.f1_score(ground_truth, predictions)

    print "prec: " + str(prec)
    print "rec: " + str(rec)
    print "f1: " + str(f1)
    
    return f1

# Build linear SVM classifier, l1 regularization to perform implicit feature selection
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, Y_train)
model = SelectFromModel(lsvc, prefit=True)

features_selected = [elem for selected, elem in zip(model.get_support(), data_loader.get_feature_names()) if selected]
print "Feature names:"
print features_selected

Y_pred = lsvc.predict(X_test)

score = custom_scorer(Y_test, Y_pred)


## Conclusions:
#
# prec: 0.714285714286
# rec: 0.111607142857
# f1: 0.19305019305
#
# By L1 regularize feature specific weights, we achieve a feature selection since the weights of some features turn to zero. By inspecting 
# lsvc.coef_ we can see that International_plan and "Number of service calls" seem most significant. However, with only a f1 score of 0.19,
# one may not put too much weight of this conclusion.