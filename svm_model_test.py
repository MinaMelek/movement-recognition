#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:22:39 2018

@author: minamelek
"""

import pandas as pd
import numpy as np
#from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import LabelEncoder
import pickle
import cv2
import os
import argparse
import logging
import sys
import time
#from tf_pose import common
#import numpy as np
#import pandas as pd
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from UsedFunctions import manipulate, Get_Coords, Calculate_D, Calculate_L, Calculate_PCM, Calculate_TCM, Calculate_R, Add_Features_To_dataframe

total_timing=time.time()             # For time computing

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

""" write your own path here """
video_path = './newData/3018.mp4'
# load the model from disk
model_path = 'SVM_model.sav'
with open(model_path, 'rb') as handle:
    loaded_model = pickle.load(handle)
    classes = pickle.load(handle)
    Y = pickle.load(handle) # labels of the data used for training
parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
parser.add_argument('--video-path', type=str, default=video_path,
                    help='if provided, select videos in that directory. ')
parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
parser.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')
#parser.add_argument('--mode', type=str, default='test',
#                    help='if provided, choose the run mode to be eather train or test. default=test ')
args = parser.parse_args()

logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
#w, h = model_wh(args.resolution)
#e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
w, h = model_wh(args.resize)
if w == 0 or h == 0:
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
else:
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
"""        
newpath=os.getcwd()+'/Output'+'/'
if not os.path.exists(newpath):
    os.makedirs(newpath)
OUTPUT_PATH=newpath+'/'+'Humans_'+args.mode+'.csv'
"""
OUTPUTS = []

video_timing=time.time()             # For time computing
print("Reading video file ", video_path)

cap = cv2.VideoCapture(video_path)
ret_val = True
if cap.isOpened() is False:
    print("Error opening video stream or file")
count = 0
while ret_val:
    
    print ("Frame No: " , count)
    
    begin_counting=time.time()             # For time computing
    ret_val, image = cap.read()
    if not ret_val:
        break
    count += 1
    if count%2:
        print('skipping frame ')
        continue
        
    #image = cv2.imread("Output/Frame No_320.jpg")
    start=time.time()
    #cv2.imwrite("Output/"+h_name+"_Frame_No_"+str(count)+".jpg",image)
    print('Time taken for writing a Frame is {:.3f} ms'.format((time.time()-start)*1000))
    
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    #humans = e.inference(image)
    print (humans)
    if len(humans)==0:
        print("Inconvenient frame")
        continue
    #for P in range(len(humans)):
    Co_ordinates=manipulate(humans, frame=count-1, mode='test')     # Computes the Co-ordinates from the given data by O-Nect
    
    print('Time taken for Co-ordinate calculations for {} Frames is {:.3f} ms'.format(len(Co_ordinates),(time.time()-begin_counting)*1000))
    
    # Extracting X, Y Coordinates
    start=time.time()
    X_Coords, Y_Coords= Get_Coords(Co_ordinates)
    print('Time taken for Co-ordinate extractions for {} Frames is {:.3f} ms'.format(len(Co_ordinates),(time.time()-start)*1000))
    
    start=time.time()
    PCM_Frames= Calculate_PCM(X_Coords, Y_Coords)
    print('Time taken for PCM for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
    
    start=time.time()
    TCM_x, TCM_y= Calculate_TCM(PCM_Frames)
    print('Time taken for TCM for {} Frames is {:.3f} ms'.format(len(TCM_x),(time.time()-start)*1000))
    
    start=time.time()
    L= Calculate_L(TCM_x, TCM_y, PCM_Frames)
    print('Time taken for L features for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
    
    start=time.time()
    D1, D2, D3 = Calculate_D(PCM_Frames, TCM_x, TCM_y, 'Degrees')
    print('Time taken for D1, D2, D3 features for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
    
    start=time.time()
    R=Calculate_R(PCM_Frames)
    print('Time taken for R feature for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
    
    start=time.time()
    out=Add_Features_To_dataframe(Co_ordinates, PCM_Frames, TCM_x, TCM_y, L, R, D1, D2, D3)
    print('Time taken for adding features to dataframe for {} Frames is {:.3f} ms'.format(len(Co_ordinates),(time.time()-start)*1000))
    
    print('Time taken for the whole file of {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-begin_counting)*1000))
    #count = count + 1
    OUTPUTS.append(out)
    #r = pd.concat(OUTPUTS)
    #r.to_csv(OUTPUT_PATH, index=False)

result = pd.concat(OUTPUTS)
#result.to_csv('test.csv', index=False)
#result = pd.read_csv('test.csv') #read data
result = result.fillna(0)
result = result.replace('nan', 0)
#dropping usless columns and columns with all their values are 0
dropped = ['Human', 'V_title', 'Frame', 'L Foot_x_PCM', 'L Foot_y_PCM', 'R Foot_L',	'L Foot_L',	'Head & Neck_R',	'Trunk_R',	'R Upper Arm_R',	'L Upper Arm_R',	'R Forearm_R',	'L Forearm_R',	'R Hand_R',	'L Hand_R',	'R Thigh_R',	'L Thigh_R',	'R Shank_R',	'L Shank_R',	'R Foot_R',	'L Foot_R']
result=result.drop(columns=dropped)
array = result.values #conver result to array
X = np.array(array)
#feature scaling 
scaler = StandardScaler()
X = scaler.fit_transform(X)
"""
#pca
pca = PCA(n_components=20)
pca.fit(X)
X = pca.transform(X)
"""
predicted = loaded_model.predict(X)
print('Predicted value is ', predicted)
print('The predicted person is', classes[max(set(predicted), key=list(predicted).count)])
print('Time taken for the whole {} file is {:.3f} minuts'.format(os.path.split(video_path)[1], (time.time()-video_timing)/60))
