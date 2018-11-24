#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:43:48 2018

@author: minamelek
"""
import cv2
import os
import argparse
import logging
import sys
import time

#from tf_pose import common
#import numpy as np
import pandas as pd
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
videos_path = './DecimalSpace-DB'


parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
parser.add_argument('--video-directory', type=str, default=videos_path,
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
parser.add_argument('--mode', type=str, default='test',
                    help='if provided, choose the run mode to be eather train or test. default=test ')
args = parser.parse_args()

logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
#w, h = model_wh(args.resolution)
#e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
w, h = model_wh(args.resize)
if w == 0 or h == 0:
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
else:
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        
newpath=os.path.join(os.getcwd(),'Output')
if not os.path.exists(newpath):
    os.makedirs(newpath)
OUTPUT_PATH=os.path.join(newpath,'Humans_'+args.mode+'.csv')

OUTPUTS = []

def RUN(file_path, h_name, v_name=''):
    video_timing=time.time()             # For time computing
    print("Reading video file ", file_path)
    
    cap = cv2.VideoCapture(file_path)
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
        if len(humans)==0 or (args.mode=='train' and len(humans)>1):
            print("Inconvenient frame")
            continue
        Co_ordinates=manipulate(humans, h_name, v_name, count-1, args.mode)     # Computes the Co-ordinates from the given data by O-Nect
        
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
        r = pd.concat(OUTPUTS)
        r.to_csv(OUTPUT_PATH, index=False)
    print('Time taken for the whole {} file is {:.3f} minuts'.format(v if 'v' in locals() else f, (time.time()-video_timing)/60))
    return OUTPUTS

print("Initiating {} mode...".format(args.mode))

# optional: to activate un comment the firs section in the for loop
valid_videos = [".avi",".mp4",".mov",".mkv",".rmvb", ".dv", ".ts"] 
for f in os.listdir(args.video_directory):
    h_name=os.path.splitext(f)[0]
    if os.path.splitext(f)[1] == '': # is it a directory?
        for v in os.listdir(os.path.join(args.video_directory,f)):
            #"""
            # for including only video formates
            ext = os.path.splitext(v)[1]
            if ext.lower() not in valid_videos:
                continue
            #"""
            OUTPUTS = RUN(os.path.join(args.video_directory,f, v), str(h_name), str(os.path.splitext(v)[0]))
        if 'v' in globals(): del v 
    else:
        #"""
        # for including only video formates
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_videos:
            continue
        #"""
        OUTPUTS = RUN(os.path.join(args.video_directory,f), str(h_name))
result = pd.concat(OUTPUTS)
result.to_csv(OUTPUT_PATH, index=False)
print('Total time taken is {:.3f} minuts'.format((time.time()-total_timing)/60))
