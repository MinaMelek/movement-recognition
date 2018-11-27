#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:31:16 2018

@author: minamelek
"""

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

from tf_pose import common
import numpy as np
import pandas as pd
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from UsedFunctions import manipulate, Get_Coords, Get_Mass, Calculate_D, Calculate_L, Calculate_PCM, Calculate_TCM, Calculate_R, Add_Features_To_dataframe

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


""" write your own path here """
videos_path = './Videos'
newpath=os.getcwd()+'/Output'+'/'
if not os.path.exists(newpath):
    os.makedirs(newpath)
OUTPUT_PATH=newpath+'/'+'Humans.csv'
# optional: to activate un comment the firs section in the for loop
valid_videos = [".avi",".mp4",".mov",".mkv",".rmvb", ".dv", ".Ts"] 
for f in os.listdir(videos_path):
    """
    # for including only video formates
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_videos:
        continue
    """
    print("Reading video file ", os.path.join(videos_path,f))
    h_name = os.path.splitext(f)[0]
    """
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default=os.path.join(videos_path,f))
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    """
    cap = cv2.VideoCapture(os.path.join(videos_path,f))
    ret_val = True
    count = 0
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    OUTPUTS = []
    while cap.isOpened() and ret_val:
        begin_counting=time.time()             # For time computing
        ret_val, image = cap.read()
        if not ret_val:
            break
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file 
        os.remove("frame%d.jpg" % count, image)
        humans = e.inference(image)
        
        Co_ordinates=manipulate(humans, h_name+'_'+str(count))     # Computes the Co-ordinates from the given data by O-Nect
        if Co_ordinates==None:
            print("No humans in this frame")
            continue
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
        
        print('Time taken the for whole file of {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-begin_counting)*1000))
        OUTPUTS.append(out)
        
OUTPUTS.to_csv(OUTPUT_PATH)