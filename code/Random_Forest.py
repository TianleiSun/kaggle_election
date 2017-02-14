
# coding: utf-8

# In[2]:

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


# In[2]:

import os
import sys
import subprocess

if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')


# In[4]:

X_train = data_train[:, 1:len(data_train[0])-2]
y_train = data_train[:, len(data_train[0])-1]
X_test = data_test[:, 1:len(data_test[0])-1]
y_test = data_test[:, len(data_test[0])-1]


# In[ ]:

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

classifiers = [DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()]

# iterate over classifiers
name = "Random Forest"
depth = [10,40,70,100,130,160,190,220,250,280,310]
n_list = [10,40,70,100,130,160,190,220]  
score_list = []
for ne in n_list:
    for d in depth:
        clf = RandomForestClassifier(max_depth = d, n_estimators = ne)
        testscore = np.mean(cross_val_score(clf, X_train, y_train, cv = 5))
        score_list.append((testscore,d,ne))
        print(testscore,d,ne)


# In[ ]:

# iterate over classifiers
name = "Random Forest"
depth = [10,40,70,100,130,160,190,220,250,280,310,340,370,400,500,600]
n_list = [10,40,70,100,130,160,190,220,300,400,500,600]  
score_list = []
for ne in n_list:
    for d in depth:
        clf = RandomForestClassifier(max_depth = d, n_estimators = ne)
        testscore = np.mean(cross_val_score(clf, X_train, y_train, cv = 5))
        score_list.append((testscore,d,ne))
        print(testscore,d,ne)


# In[7]:

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

classifiers = [DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()]

# iterate over classifiers
name = "Random Forest"
n_list = [1,4,7,10,15]  
score_list = []
for ne in n_list:
    clf = RandomForestClassifier(max_depth = 130, n_estimators = 220, min_samples_leaf = ne)
    testscore = np.mean(cross_val_score(clf, X_train, y_train, cv = 5))
    score_list.append((testscore,ne))
    print(testscore,ne)

