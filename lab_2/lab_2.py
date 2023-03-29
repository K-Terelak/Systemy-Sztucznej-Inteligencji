#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random


class DataProcessing:
    data = None

    def __init__(self, d):
        self.data = d
        pd.set_option('display.max_rows', len(self.data))

    def shuffle(self):
        for i in range(len(self.data) - 1, 0, -1):
            j = random.randint(0, i)
            self.data.iloc[i], self.data.iloc[j] = self.data.iloc[j], self.data.iloc[i]
        return self.data

    def split(self):
        split = int(len(self.data) * 0.7)
        listTrain = self.data.iloc[:split, :]
        listVal = self.data.iloc[split:, :]
        return listTrain, listVal

    def normalize(self):
        listCopy = self.data.copy()
        values = listCopy.select_dtypes(exclude="object")  # no variety column
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = listCopy.loc[:, column]
            minimum, maximum = min(data), max(data)
            for row in range(0, len(listCopy), 1):
                value = (listCopy.at[row, column] - minimum) / (maximum - minimum)  # (x-min)/(max-min)
                listCopy.at[row, column] = value
        self.data = listCopy


class KNN:
    @staticmethod
    def minkowskiDistance(vec1, vec2, m):
        distance = 0
        for i in range(len(vec2) - 1):
            distance += abs(vec2[i] - vec1[i]) ** m
        distance = float(distance ** (1 / m))
        return distance

    @staticmethod
    def sortByDistance(testIris, trainBase, m):
        distances = []
        for i in range(len(trainBase)):
            distances.append(KNN.minkowskiDistance(testIris, trainBase.iloc[i], m))
        return KNN.sorting(trainBase, distances)

    @staticmethod
    def sorting(trainList, distances):
        trainBase = trainList.copy()
        for i in range(len(trainBase)):
            ready = True
            print(".", end="")
            for j in range(len(trainBase) - i - 1):
                if distances[j] > distances[j + 1]:
                    distances[j], distances[j + 1] = distances[j + 1], distances[j]
                    trainBase.iloc[j], trainBase.iloc[j + 1] = trainBase.iloc[j + 1], trainBase.iloc[j]
                    ready = False
            if ready:
                break
        return trainBase

    @staticmethod
    def clustering(valBase, trainBase, k, m):
        corrected = 0
        n = len(valBase)
        fail = {"Setosa": 0, "Virginica": 0, "Versicolor": 0}
        types = {"Setosa": 0, "Virginica": 0, "Versicolor": 0}
        for irisId in range(n):
            testIris = valBase.iloc[irisId]
            testIrisVariety = valBase.iloc[irisId].variety
            types[testIrisVariety] += 1
            classes = {"Setosa": 0, "Virginica": 0, "Versicolor": 0}
            print("\n({}/{})".format(irisId + 1, n))
            # sort elements by distance
            trainBase = KNN.sortByDistance(testIris, trainBase, m)
            # get k nearest
            for i in range(0, k, 1):
                classes[trainBase.iloc[i].variety] += 1
            # predict
            aiIris = max(classes, key=classes.get)
            if aiIris == testIrisVariety:
                corrected += 1
            else:
                fail[testIrisVariety] += 1
        # show results
        print("\n\tResults:\n"
              "k = {}\nm = {}".format(k, m))
        k = 0
        for key, value in types.items():
            all = list(types.values())[k]
            this = list(fail.values())[k]
            diff = all - this
            print("{} - {}/{} - {:.2f}%".format(key, diff, all, diff / all * 100))
            k += 1
        accuracy = corrected / len(valBase) * 100
        print("Accuracy - {:.2f}%".format(accuracy))
        return accuracy


dataSet = pd.read_csv('iris.csv')
dp = DataProcessing(dataSet)
acc = []
for k in range(2, 6, 1):
    dp.shuffle()
    dp.normalize()
    training, values = dp.split()
    acc.append(KNN.clustering(values, training, k, 2))
for k in range(2, 6, 1):
    print("k = {} - {:.2f}% accuracy.".format(k, acc[k - 2]))

