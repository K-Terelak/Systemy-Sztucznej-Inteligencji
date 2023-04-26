#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import math

#Zadanie 1 - klasa
class DataProcessing:
    @staticmethod
    def shuffle(X):
        for i in range(len(X)-1,0,-1):
            j = random.randint(0,i)
            X.iloc[i], X.iloc[j] = X.iloc[j], X.iloc[i]
        return X
    @staticmethod
    def normalize(n):
        k = n.drop("variety", axis=1)
        k1 = (k - k.min())/(k.max()-k.min())
        for (name,col) in k1.items():
            n[name] = col
        return n
    @staticmethod
    def split(X, x):
        if x >= 10 or x < 0:
            raise ValueError('Niepoprawne x (musi być < 1)')
        return X[:math.ceil(len(X)*x)],X[math.ceil(len(X)*(1-x)):]

#Zadanie 2 - klasa
class bayes:
    def triangle(x, mean, std):
        left = mean - math.sqrt(6) * std
        right = mean + math.sqrt(6) * std
        if x < left or x > right:
            return 0
        if x <= mean:
            return (x - left) / ((mean - left) * std)
        else:
            return (right - x) / ((right - mean) * std)

    #klasyfikacja
    def classify(train, sample):
        imiona = train.variety.unique()
        classes = []
        for imie in imiona:
            classes += [train[train['variety'] == imie]]
            del classes[-1]['variety']
        classes_triangle = []
        for classy in classes:
            mean = []
            std = []
            triangle = []
            for (imie, data) in classy.items():
                mean += [np.mean(data.values)]
                std += [np.std(data.values)]
                triangle += [bayes.triangle(sample[imie], mean[-1], std[-1])]
            classes_triangle += [math.prod(triangle)]
        return imiona[classes_triangle.index(max(classes_triangle))]


# In[2]:


#Zadanie 2 - test
seeds = pd.read_csv('iris.csv')
seeds = DataProcessing.shuffle(seeds)
seeds = DataProcessing.normalize(seeds)
seedtrain, seedvalid = DataProcessing.split(seeds, 0.7)
correct = 0
for i in range(0,len(seedvalid)):
    sample = seedvalid.iloc[i].drop('variety').to_dict()
    if seedvalid.iloc[i].variety == bayes.classify(seedtrain, sample):
        correct += 1
accuracy = correct / len(seedvalid.index) * 100
print("Dokladnosc (znormalizowana) -", accuracy, "%")


# In[ ]:




