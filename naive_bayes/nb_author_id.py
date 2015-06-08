#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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




#########################################################
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB() #create classifer

t0 = time()
clf.fit(features_train,labels_train) #fit the classifier from the train data
print "training time:", round(time()-t0, 3), "s"

t1 = time()
accuracy = clf.score(features_test, labels_test)
print "predicting time:", round(time()-t1, 3), "s"

print "Accuracy from GaussianNB = ", accuracy

#########################################################

#Want to figure how to print an image
# from class_vis import prettyPicture, output_image
# prettyPicture(clf, features_test,labels_test)
# output_image("result.png", "png", open("result.png","rb").read())