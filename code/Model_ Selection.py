
# coding: utf-8

# In[1]:

import csv
import numpy as np

with open('train_2008.csv', 'r') as dest_f:
	data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"') 
	data1 = [data for data in data_iter]

data_train = np.asarray(data1[1:], dtype = 'i8')

with open('test_2008.csv', 'r') as dest_f:
	data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"') 
	data2 = [data for data in data_iter]

data_test = np.asarray(data2[1:], dtype = 'i8')

X_train = data_train[:, 1:len(data_train[0])-2]
y_train = data_train[:, len(data_train[0])-1]


# In[16]:

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


names = ["Nearest\n Neighbors", "Decision\n Tree", "Random\n Forest", "Neural\n Net", "AdaBoost"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()]

datasets = [(X_train, y_train)]


# iterate over classifiers
performance = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train) 
    score = cross_val_score(clf, X_train, y_train, cv=5)
    performance.append(np.mean(score))


# In[17]:

y_pos = np.arange(len(names)) 
plt.bar(y_pos, performance, width = 0.5, linewidth = 1, align='center', alpha=0.8, orientation = 'vertical')
plt.xticks(y_pos, names)
plt.ylabel('Accuracy')
plt.title('Performance of models')
plt.show()


# In[ ]:



