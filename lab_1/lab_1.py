#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.Wczytaj bazę danych Iris, dokonaj analizy bazy oraz wygeneruj podstawowe informacje o niej i poszczególnych atrybutach (np.: za pomocą wykresów). 
#2. Utwórz klasę ProcessingData, w której zaimplementowane zostałą statyczne metody do normalizacji oraz tasowania zbioru, podziału bazy na zbiór treningowy/walidacyjny. 


# In[2]:


import pandas as pd
import seaborn as sb
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


# In[4]:


# load dataSet
dataSet = pd.read_csv('iris.csv')


# In[10]:


dataSet.describe()
sb.pairplot(dataSet, hue="variety")


# In[7]:


# shuffle, normalize, split
dp = DataProcessing(dataSet)
dp.shuffle()
dp.normalize()
training, values = dp.split()


# In[8]:


# Training data
training


# In[9]:


# Data
values


# In[ ]:




