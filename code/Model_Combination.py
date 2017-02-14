import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost

import os
import sys
import subprocess

if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')

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

rfc = xgboost.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=240)
abc = AdaBoostClassifier(n_estimators = 200)

kf = KFold(n_splits=5)
kf.get_n_splits(X_train)
# print kf    
# KFold(n_splits=5, random_state=None, shuffle=False)
A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

for a in A:
    for train_index, test_index in kf.split(X_train):
        # print("TRAIN:", train_index, "TEST:", test_index)
        accuracy = []
        X_train_val, X_test_val = X_train[train_index], X_train[test_index]
        y_train_val, y_test_val = y_train[train_index], y_train[test_index]
        rfc.fit(X_train_val, y_train_val)
        abc.fit(X_train_val, y_train_val)
        Y_rfc = rfc.predict(X_test_val)
        Y_abc = abc.predict(X_test_val)
        #rand1 = np.random.uniform(0, 1)
        #rand2 = random.uniform(0, 1)
        Y = Y_rfc * a + Y_abc * (1 - a)
        for i in range(len(Y)):
            if Y[i] < np.random.uniform(1, 2):
                Y[i] = 1
            else:
                Y[i] = 2
        print Y
        error = np.sum(np.abs(Y - y_test_val))*1.0/len(Y_rfc)
        accuracy.append(1 - error)
    print (np.mean(accuracy), a)

'''
a = 0.6
rfc.fit(X_train, y_train)
abc.fit(X_train, y_train)
y_test_1 = rfc.predict(X_test)
y_test_2 = abc.predict(X_test)
y_test = y_test_1 * a + y_test_2 * (1 - a)
for i in range(len(y_test)):
    if y_test[i] < np.random.uniform(1, 2):
        y_test[i] = 1
    else:
        y_test[i] = 2
with open('2008.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "PES1"])
    for i in range(len(y_test)):
        writer.writerow([i, y_test[i]])
'''