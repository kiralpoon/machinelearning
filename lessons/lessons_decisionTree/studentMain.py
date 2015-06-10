#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt

import numpy as np
import pylab as pl
from classifyDT import classify

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

features_train, labels_train, features_test, labels_test = makeTerrainData()

#adding taking parameter from the commend line to have different min split sample
s = sys.argv[1] #taking the first parameter
if (RepresentsInt(s) == True):
	min_sample_split = int(s)

else:
	min_sample_split = 2 #default value

### the classify() function in classifyDT is where the magic
### happens--it's your job to fill this in!
clf = classify(features_train, labels_train, min_sample_split)

#getting the acuuracy
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
plt.show() #I add this line to show the image from matplotlib

#accuracy result will be shown after the show window is close
print ('Accuacy from decision tree = ', acc)

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
print 'min_sample_split used in decision tree: ', min_sample_split

