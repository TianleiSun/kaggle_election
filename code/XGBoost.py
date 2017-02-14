import csv
import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.gaussian_process import GaussianProcessClassifier
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
'''
#alpha = [0.000001, 0.0001]
nn = [100, 130,160,180,200, 210, 220]
#leaf = [200, 220, 240, 260, 280, 300]
validation_error_list = []
#for d in depth:
for n in nn:
	rfc = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
	rfc.fit(X_train, y_train)
	score = cross_val_score(rfc, X_train, y_train, cv=5)
	validation_error_list.append((np.mean(score), n))
	print (np.mean(score), n)
'''

# fit model no training data
NE = [200, 220, 240, 260, 280, 300]
for ne in NE:
	model = xgboost.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=ne)
	model.fit(X_train, y_train)
	score = cross_val_score(model, X_train, y_train, cv=5)
	print (np.mean(score), ne)

NE = [200, 220, 240, 260, 280]
for ne in NE:
    model = xgboost.XGBClassifier(max_depth=10, learning_rate=1, n_estimators=ne)
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_train, y_train, cv=5)
    print(ne, np.mean(score))