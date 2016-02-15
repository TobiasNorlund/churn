
import numpy as np
from sklearn.linear_model import *
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

lsvc = LogisticRegression(penalty="l1").fit(X_train, Y_train)
model = SelectFromModel(lsvc, prefit=True)

features_selected = [elem for selected, elem in zip(model.get_support(), data_loader.get_feature_names()) if selected]
print "Feature coef:"
print lsvc.coef_
print "Feature names:"
print features_selected

Y_pred = lsvc.predict(X_test)

score = custom_scorer(Y_test, Y_pred)


## Conclusions:
#
# A Logistic Regression classifier can also be L1 regularized to perform implicit feature selection. Only some state features' weights turned zero in 
# this case but by inspecting the weight values features like International_plan and "number of service calls" once again shows up as most significant
# 
# >>> names_arr[top[0]]
# array(['NJ', 'MT', 'TX', 'SC', 'CA', 'Internatioal_plan',
#        'number customer service calls'], 
#        dtype='|S29')

# prec: 0.540540540541
# rec: 0.267857142857
# f1: 0.358208955224