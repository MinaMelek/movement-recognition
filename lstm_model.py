#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:18:14 2018

@author: minamelek
"""
# In[1] Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import pickle
from random import shuffle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils.np_utils import to_categorical
from UsedFunctions import modelBuild
# In[2] Methods
def ReadData(seq):
    """ ----- Very important to note the pathes ----- """
    # getting training data
    df = pd.read_csv(os.path.join('.','newData','Humans_train.csv')) #read data
    df = df.append(pd.read_csv(os.path.join('.','Humans_train_lite.csv')).replace('P1', 'P11'))
    df = df.append(pd.read_csv(os.path.join('.','newData','Humans_train_added.csv')))
    #editing on data
    df = df.fillna(0) #conver NaN values to zeros
    
    dropped = ['Frame', 'L Foot_x_PCM', 'L Foot_y_PCM', 'R Foot_L',	'L Foot_L',	'Head & Neck_R',	'Trunk_R',	'R Upper Arm_R',	'L Upper Arm_R',	'R Forearm_R',	'L Forearm_R',	'R Hand_R',	'L Hand_R',	'R Thigh_R',	'L Thigh_R',	'R Shank_R',	'L Shank_R',	'R Foot_R',	'L Foot_R']
    #dropping usless columns and columns with all their values are 0
    df = df.drop(columns=dropped)
    
    array = df.values #conver df to array
    X = np.array(array)
    target = array[:, 0] #split target data
    Vtitle = df['V_title'].values.astype('str')
    
    labelencoder = LabelEncoder()
    target = labelencoder.fit_transform(target)
    Vtitle = labelencoder.fit_transform(Vtitle)
    # get unique labels 
    l = list(set(df.Human))
    unique_labels = dict(zip(labelencoder.fit_transform(l), l))
    
    X = np.delete(X, np.s_[:2], 1) #drop target data 
    #x = np.float64(X)
    Y = np.array(target)
    Y=Y.astype('int') # issue
    
    #feature scaling 
    scaler = MinMaxScaler()
    X = scaler.fit_transform(np.float64(X))
    
    Data_X = []
    Data_Y = []
    Num = 0
    ind = 0
    for cLen in np.bincount(Vtitle):
        for i in range(1 + (cLen // seq)):
            x = []
            for j in range(seq):
                if (seq*i + j) >= cLen:
                    x.append([0]*X.shape[1])
                else:
                    x.append(list(X[Num + seq*i + j]))
                    y = Y[Num + seq*i + j]
            Data_X.append(x)
            Data_Y.append(y) 
        Num += cLen
        ind += 1
    Data_Y = to_categorical(Data_Y) # convert to one hot
    return np.array(Data_X, dtype='float32'), np.array(Data_Y, dtype='float32'), unique_labels
# In[3] Begining of Training
beg = time.time()
# Variables
sequence_size = 5
batch_size = 64
epochs = 200
lstm_out = [128, 256, 512, 1024, 64]
dense_layer = 128
# In[4] Data manipulation

#reading data 
Data_X, Data_Y, classes = ReadData(sequence_size)
# shuffling data
Z= list(zip(Data_X, Data_Y))
shuffle(Z)
Data_X, Data_Y = zip(*Z)
Data_X, Data_Y = np.array(Data_X), np.array(Data_Y)
X_train, X_test, y_train, y_test = train_test_split(Data_X, Data_Y, test_size=0.20, random_state=42)
# In[5] Training

# Building the model
model = modelBuild(X_train.shape[1:], lstm_out, [dense_layer, len(classes)])
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Saving weights of the model after every epoch in a given directory
modelspath=os.path.join(os.getcwd(),'ModelData')
if not os.path.exists(modelspath):
    os.makedirs(modelspath)
# Preparing callbacks during training
filepath=os.path.join(modelspath,'weights-{epoch:02d}-{val_acc:.3f}.hdf5')
earlystop = EarlyStopping(monitor='val_loss', patience=125, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True)
# using tensorbard to visualize the training and view learning curves
# to know when to stop and choose which epoch as the best, while the code is running 
# run the following command in your terminal while pointing to the script directory
tbCallBack = TensorBoard(log_dir='log', histogram_freq=0, write_graph=False, write_images=True)
model.load_weights(os.path.join(modelspath,'weights-113-0.94.hdf5'))
# training the model and using validation data to validate the parameters (overfitting and so on..)
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs = epochs, 
          batch_size = batch_size, shuffle=True, verbose = 2, 
          callbacks=[checkpoint,tbCallBack])#,earlystop])
# In[6] Save necessary files

model.save(os.path.join(modelspath,'my_model.hdf5'))
model.save_weights(os.path.join(modelspath,'my_model_weights.hdf5'))
with open(os.path.join(modelspath,'model_architecture.json'), 'w') as f:
    f.write(model.to_json())
# Saving some variables that might be used during testing
filename = os.path.join(modelspath, 'LSTM_model_variables.sav')
with open(filename, 'wb') as handle:
    pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sequence_size, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(batch_size, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# In[6] Evaluating

loss, acc = model.evaluate(X_test, y_test)
print()
print("Test accuracy = ", acc)

# Prediction
pred = model.predict(X_test)
#print(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1)))
y_actu = pd.Series(y_test.argmax(axis=1), name='Actual')
y_pred = pd.Series(pred.argmax(axis=1), name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)
print('Total time taken is {:.3f} min'.format((time.time()-beg)/60))