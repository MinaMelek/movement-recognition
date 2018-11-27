# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 01:52:00 2018

@author: Ahmed Raafat

"""

import pandas as pd
import numpy as np

def create_dict(p1_id, p2_id, x1, x2, y1, y2):
    '''
    Loops over and appends the value of the co-ordinates for its own id, so the values
    with the same id won't be hurt as they are the same number, therefore we arent duplicating the 
    id. and finally returns the dictionary which contains the part id with its co-ordinates
    
    '''    
 
    Pose_keypoints={0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[],
               10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}

    for i in range(len(p1_id)):                   
        #if (p1_id[i]==1) & (p2_id[i]==2) & i>0:
        #    j+=1
        #    Pose_keypoints={0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[],
        #       10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}
            
        Pose_keypoints[p1_id[i]]=[x1[i], y1[i]]           
        Pose_keypoints[p2_id[i]]=[x2[i], y2[i]]
        #print('p1_id={}, p2_id={}'.format(p1_id[i], p2_id[i]))
   
    return Pose_keypoints
       
def manipulate(Pose_Persons, directory='No'): 
    '''this takes a dictionary  of parts in the image along with the coordinates 
        and outputs a csv file to a given directory containing the co-ordinate of each part with 
        a label'''
      
    col=['Frame Number', 'Nose_x','Nose_y','Neck_x','Neck_y','RShoulder_x','RShoulder_y',
         'RElbow_x','RElbow_y','RWrist_x','RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y',
         'LWrist_x','LWrist_y','RHip_x','RHip_y','RKnee_x','RKnee_y','RAnkle_x','RAnkle_y',
         'LHip_x','LHip_y','LKnee_x','LKnee_y','LAnkle_x','LAnkle_y','REye_x','REye_y',
         'LEye_x','LEye_y','REar_x','REar_y','LEar_x','LEar_y']
    x_data=[]
    y_data=[]
    all_data=[]
    keys=list(Pose_Persons.keys())      # A list of the frame numbers
    for i in range(len(keys)):
        x=[]
        y=[]
        P=list(Pose_Persons[i].values())  # Gets all the values of frame number i
       
        for j in range(18):     # Loops over 18 features which are the same as in col but divided by 2
            if P[j] == []:
                x.append('nan') # Appends nan if the value is nothing so we wont have problems with
                y.append('nan') # the calculations when adding, subtracting, multiplying
            else:
                x.append(P[j][0]) # Appends each feature
                y.append(P[j][1])
        x_data.append(x)          # Appends whole features (18) in 1 list
        y_data.append(y)
    
    for i in range(len(keys)):
        all_data.append(keys[i])  # Appends all data together in shape of x1, y1, x2, y2
        [[all_data.append(k), all_data.append(j)] for k,j in zip(x_data[i], y_data[i])]
        
    all_data=np.reshape(np.array(all_data), (len(keys),37))    # reshapes into 37 columns 1 for frame number and 36 for all features in x and y
    output=pd.DataFrame(all_data, columns=col)
    #output=output.replace(0,'NA')
    if not directory =='No':
        output.to_csv(path_or_buf=directory)
    return output
    
def manip(human, directory='No'):
    col=['Human', 'Nose_x','Nose_y','Neck_x','Neck_y','RShoulder_x','RShoulder_y',
         'RElbow_x','RElbow_y','RWrist_x','RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y',
         'LWrist_x','LWrist_y','RHip_x','RHip_y','RKnee_x','RKnee_y','RAnkle_x','RAnkle_y',
         'LHip_x','LHip_y','LKnee_x','LKnee_y','LAnkle_x','LAnkle_y','REye_x','REye_y',
         'LEye_x','LEye_y','REar_x','REar_y','LEar_x','LEar_y']
    x_data=[]
    y_data=[]
    #scores=[]
    all_data=[]
    for i in range(len(human)):
        x=[]
        y=[]
        #s=[]
       # P=list(human[i].values())  # Gets all the values of frame number i
       
        for j in range(len(human[0].body_parts)):     # Loops over 18 features which are the same as in col but divided by 2
            x.append(human[i].body_parts[j].x) # Appends each feature
            y.append(human[i].body_parts[j].y)
            #s.append(human[i].body_parts[j].score)
        x_data.append(x)          # Appends whole features (18) in 1 list
        y_data.append(y)
        #scores.append(s)
    
    for i in range(len(human)):
        all_data.append("Person"+str(i+1))  # Appends all data together in shape of x1, y1, x2, y2
        [[all_data.append(k), all_data.append(l)] for k,l in zip(x_data[i], y_data[i])]
        #[[all_data.append(k), all_data.append(l), all_data.append(m)] for k,l,m in zip(x_data[i], y_data[i], scores[i])]
        
    all_data=np.reshape(np.array(all_data), (len(human),len(col)))    # reshapes into 55 columns 1 for frame number and 54 for all features in x, y and score
    output=pd.DataFrame(all_data, columns=col)
    #output=output.replace(0,'NA')
    if not directory =='No':
        output.to_csv(path_or_buf=directory)
    return output