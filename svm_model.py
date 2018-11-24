import csv 
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import time

beg = time.time()
#reading data 
df = pd.read_csv('./newData/Humans_train.csv') #read data
#editing on data
df = df.fillna(0) #conver NaN values to zeros

dropped = ['V_title', 'Frame', 'L Foot_x_PCM', 'L Foot_y_PCM', 'R Foot_L',	'L Foot_L',	'Head & Neck_R',	'Trunk_R',	'R Upper Arm_R',	'L Upper Arm_R',	'R Forearm_R',	'L Forearm_R',	'R Hand_R',	'L Hand_R',	'R Thigh_R',	'L Thigh_R',	'R Shank_R',	'L Shank_R',	'R Foot_R',	'L Foot_R']
#dropping usless columns and columns with all their values are 0
df = df.drop(columns=dropped)

array = df.values #conver df to array
X = np.array(array)
target = array[:, 0] #split target data

#Label Encoder
labelencoder = LabelEncoder()
target = labelencoder.fit_transform(target)
# get unique labels 
l = list(set(df.Human))
unique_labels = dict(zip(labelencoder.fit_transform(l), l))

X = np.delete(X, 0, 1) #drop target data 
#x = np.float64(X)
Y = np.array(target)
Y=Y.astype('int') # issue

#feature scaling 
scaler = StandardScaler()
X = scaler.fit_transform(np.float64(X))
"""
#pca
pca = PCA(n_components=20)
pca.fit(X)
X = pca.transform(X)
"""
#train test split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#svm model
clf = svm.SVC(kernel='poly', degree=3, gamma='auto', coef0=1.9)

clf.fit(X_train, y_train)

filename = 'SVM_model.sav'
with open(filename, 'wb') as handle:
    pickle.dump(clf, handle)
    pickle.dump(unique_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

predicted = clf.predict(X_test)

#print accuracy
print("accuracy: {}%".format(accuracy_score(y_test, predicted)*100))
print('Total time taken is {:.3f} ms'.format((time.time()-beg)*1000))
