#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:11:12 2018

@author: M.Mohy
"""

import numpy as np
import operator
import faceRecognition
count=1
labelVector =[]
for x in range(40): #constructing the classes of each instance
   for j in range(5):
       labelVector.append(count)
   count= count+1

def getAccuracy(testSet, label_vector):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] is label_vector[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def Knn(trainingSet,label_vector,test,k):
    distances =[]
    for x in range(len(trainingSet)):
        distance = np.linalg.norm(np.subtract(test,trainingSet[x]))
        distances.append((label_vector[x],distance))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    classes ={}
    for x in range(len(neighbors)):
        response= neighbors[x][0]
        if response in classes:
            classes[response] += 1
        else:
            classes[response] = 1
    sortedVotes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
list_of_predictions=[]
k=5
for x in range(200):
   list_of_predictions.append(Knn(faceRecognition.training_data_matrix,labelVector,faceRecognition.test_data_matrix[x],k))
print(getAccuracy(list_of_predictions,labelVector),"%")
