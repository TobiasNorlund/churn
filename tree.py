
import numpy as np
import sklearn.tree
import sklearn.cross_validation
import data_loader
from sklearn.metrics import make_scorer


# Load train data
(X_train, Y_train) = data_loader.load("Dataset/churn.data.txt", standardize=False)
(X_test, Y_test)   = data_loader.load("Dataset/churn.test.txt", standardize=False)

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

model = sklearn.tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
score = custom_scorer(Y_test, Y_pred)

# Visualize graph tree
from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO() 
sklearn.tree.export_graphviz(model, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("tree.pdf") 

## Conclusions:

# :: prec, recall, f1 ~ 70%

# By inspecting the tree, dim 56 (total day minutes) seems to be chosen often (root split and many child splits).
#
# X_train[Y_train,56].mean() = 250
# X_train[1-np.array(Y_train),56].mean() = 176
#
# => Seems to be a big difference there, which makes the split viable
#
# Other split dimensions are: number customer service calls, number vmail messages
