#!/usr/bin/python

""" 
this is the code to accompany the Lesson 2 (SVM) mini-project

use an SVM to identify emails from the Enron corpus by their authors

Sara has label 0
Chris has label 1

"""

import sys
from time import time
from class_vis import prettyPicture
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#These lines effectively slice the training dataset down 
#to 1% of its original size, tossing out 99% of the training data.

#comment below code to train with full dataset 
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

#########################################################

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import copy
import numpy as np


#Training with a linear kernel
#clf = SVC(C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
#training with a rbf kernel

C_var = 10
for x in range(0, 4): #run 4 times
	clf = SVC(C=C_var, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)


	t0 = time()
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	print "C = ",C_var,"training time:", round(time()-t0, 3), "s"

	from sklearn.metrics import accuracy_score
	t1 = time()
	acc = accuracy_score(pred, labels_test)
	print "C = ",C_var,"predicting time:", round(time()-t1, 3), "s"

	print ("Accuracy from SVM = ", acc)

# prettyPicture(clf,features_test,labels_test)
# plt.show()
	C_var = C_var*10

#########################################################


