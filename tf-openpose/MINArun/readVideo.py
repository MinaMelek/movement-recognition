import cv2
import os
""" write your own path here """
videos_path = '/home/minamelek/Documents/Work/BrainWise/pose_rec/videos/'
vidcap = cv2.VideoCapture('')
# vidcap.set(cv2.CAP_PROP_POS_MSEC,10000)      # just cue to 20 sec. position
# success,image = vidcap.read()
# if success:
# 	cv2.imwrite("frame60sec.jpg", image) 
# 	for i in range (0,100):
# 		vidcap.set(cv2.CAP_PROP_POS_MSEC,10000 )      # just cue to 20 sec. position
# 		success,image = vidcap.read()
# 		cv2.imwrite("frame%d.jpg"% (i), image) 


success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1