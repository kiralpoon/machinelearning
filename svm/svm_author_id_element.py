#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#These lines effectively slice the training dataset down 
#to 1% of its original size, tossing out 99% of the training data.

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

#########################################################
from class_vis import prettyPicture
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import copy
import numpy as np


#Training with a linear kernel
#clf = SVC(C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
#training with a rbf kernel
clf = SVC(C=10000.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)


t0 = time()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "training time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
t1 = time()
acc = accuracy_score(pred, labels_test)
print "predicting time:", round(time()-t1, 3), "s"

print ("Accuracy from SVM = ", acc)


#trying to get answer for element 10
answer = pred[10]
if(answer==1):
	result = "Chris"
else:
	result = "Sara"
print "answer for element 10: ", answer, "which is ",result

answer = pred[26]
if(answer==1):
	result = "Chris"
else:
	result = "Sara"
print "answer for element 26: ", answer, "which is ",result

answer = pred[50]
if(answer==1):
	result = "Chris"
else:
	result = "Sara"
print "answer for element 50: ", answer, "which is ",result


# prettyPicture(clf,features_test,labels_test)
# plt.show()


#########################################################


