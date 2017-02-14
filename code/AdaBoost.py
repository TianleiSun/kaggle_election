import os
import sys
import subprocess

if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')

import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


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
X_test = data_test[:, 1:len(data_test[0])-1]


est_list = range(50, 300, 10)
scores = []
for est in est_list:
    clf = AdaBoostClassifier(n_estimators = est)
    #clf.fit(X_train, y_train)
    score = cross_val_score(clf, X_train, y_train, cv = 5)
    scores.append((np.mean(score), est))
    print(np.mean(score), est)


