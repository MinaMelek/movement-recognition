
# coding: utf-8

# In[23]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import warnings
warnings.filterwarnings('ignore')

######################################## List of Functions #########################################
def Get_Coords(File):
    x = File.iloc[:, [i for i in range(1,len(File.columns)) if i%2 == 1]]
    y = File.iloc[:, [i for i in range(1,len(File.columns)) if i%2 == 0]]
    return x,y

def Get_Mass(Weight, Gender='Male'):
    '''
    Computes the mass for a given Gender according to the percentages in the CM paper
    Assuming it is a Male if there is no Gender given
    
    '''
    
    Male_Mass={'Head & Neck':6.94/100, 'Trunk':43.46/100, 'Upper Arm':2.71/100, 
           'Forearm':1.62/100, 'Hand':0.61/100, 'Thigh':14.16/100, 'Shank':4.33/100, 'Foot':1.37/100}
    
    Female_Mass={'Head & Neck':6.68/100, 'Trunk':42.58/100, 'Upper Arm':2.55/100, 
           'Forearm':1.38/100, 'Hand':0.56/100, 'Thigh':14.78/100, 'Shank':4.81/100, 'Foot':1.29/100}
    
    if Gender=='Male':
        Keys=list(Male_Mass.keys())
        for i in range(len(Keys)):
            Male_Mass[Keys[i]]=Male_Mass[Keys[i]]*Weight
        return Male_Mass    

    else:
        Keys=list(Female_Mass.keys())
        for i in range(len(Keys)):
            Female_Mass[Keys[i]]=Female_Mass[Keys[i]]*Weight
        return Female_Mass  

def Calculate_PCM(X_Coords, Y_Coords, Gender ='Male'):
    '''
    We first get the length percentages according to the CM paper, then looping over each frame
    in the dataframe to compute the PCM for each required part.
    
    Note that casting to float will avoid the string compuation errors from 'nan'
    
    '''
    
    Male_Segment_Length_Percentage={'Head & Neck':50.02/100, 'Trunk':43.1/100, 'Upper Arm':57.72/100, 
               'Forearm':45.74/100, 'Hand':79/100, 'Thigh':40.95/100,'Shank':43.95/100, 'Foot':44.15/100}

    Female_Segment_Length_Percentage={'Head & Neck':48.41/100, 'Trunk':37.82/100, 'Upper Arm':57.54/100,
               'Forearm':45.59/100, 'Hand':74.74/100, 'Thigh':36.12/100,'Shank':43.52/100, 'Foot':40.14/100}
    Frame={}
    num_frames=X_Coords.shape[0]
    
    for i in range(num_frames):
        
        ############################### Data ###################################
        PCM={}
        Sample_x=X_Coords.iloc[i,:]     # Takes the ith row of the x_coords 
        Sample_y=Y_Coords.iloc[i,:]
        
        ################################ Head & Neck ###########################
        PCM['Head & Neck_x']=float(Sample_x['Neck_x'])
        PCM['Head & Neck_y']=float(Sample_y['Neck_y']) #Simply the [Neck_x, Neck_y]

        ################################# Trunk ####################################
        if Gender == 'Male':
            cm=Male_Segment_Length_Percentage['Trunk']
        else: cm=Female_Segment_Length_Percentage['Trunk']
            
        XP=(float(Sample_x['RShoulder_x'])+float(Sample_x['LShoulder_x'])/2)
        YP=(float(Sample_y['RShoulder_y'])+float(Sample_y['LShoulder_y'])/2)
        XD=(float(Sample_x['RHip_x'])+float(Sample_x['LHip_x'])/2)
        YD=(float(Sample_y['RHip_y'])+float(Sample_y['LHip_y'])/2)

        PCM['Trunk_x']=XP+cm*(XD-XP)
        PCM['Trunk_y']=YP+cm*(YD-YP)
        ################################# Upper Arm ################################
        if Gender == 'Male':
            cm=Male_Segment_Length_Percentage['Upper Arm']
        else: cm=Female_Segment_Length_Percentage['Upper Arm']
            
        XPR=float(Sample_x['RShoulder_x'])
        YPR=float(Sample_y['RShoulder_y'])
        XDR=float(Sample_x['RElbow_x'])
        YDR=float(Sample_y['RElbow_y'])

        XPL=float(Sample_x['LShoulder_x'])
        YPL=float(Sample_y['LShoulder_y'])
        XDL=float(Sample_x['LElbow_x'])
        YDL=float(Sample_y['LElbow_y'])

        PCM['R Upper Arm_x']=XPR+cm*(XDR-XPR)
        PCM['R Upper Arm_y']=YPR+cm*(YDR-YPR)
        PCM['L Upper Arm_x']=XPL+cm*(XDL-XPL)
        PCM['L Upper Arm_y']=YPL+cm*(YDL-YPL)

        ################################ Forearm  ##################################
        if Gender == 'Male':
            cm=Male_Segment_Length_Percentage['Forearm']
        else: cm=Female_Segment_Length_Percentage['Forearm']
            
        XPR=float(Sample_x['RElbow_x'])
        YPR=float(Sample_y['RElbow_y'])
        XDR=float(Sample_x['RWrist_x'])
        YDR=float(Sample_y['RWrist_y'])

        XPL=float(Sample_x['LElbow_x'])
        YPL=float(Sample_y['LElbow_y'])
        XDL=float(Sample_x['LWrist_x'])
        YDL=float(Sample_y['LWrist_y'])

        PCM['R Forearm_x']=XPR+cm*(XDR-XPR)
        PCM['R Forearm_y']=YPR+cm*(YDR-YPR)
        PCM['L Forearm_x']=XPL+cm*(XDL-XPL)
        PCM['L Forearm_y']=YPL+cm*(YDL-YPL)

        ################################ Hand #######################################
        PCM['R Hand_x']=Sample_x['RWrist_x']
        PCM['R Hand_y']=Sample_y['RWrist_y']
        PCM['L Hand_x']=Sample_x['LWrist_x']
        PCM['L Hand_y']=Sample_y['LWrist_y']

        ############################## Thigh ########################################
        if Gender == 'Male':
            cm=Male_Segment_Length_Percentage['Thigh']
        else: cm=Female_Segment_Length_Percentage['Thigh']
            
        XPR=float(Sample_x['RHip_x'])
        YPR=float(Sample_y['RHip_y'])
        XDR=float(Sample_x['RKnee_x'])
        YDR=float(Sample_y['RKnee_y'])

        XPL=float(Sample_x['LHip_x'])
        YPL=float(Sample_y['LHip_y'])
        XDL=float(Sample_x['LKnee_x'])
        YDL=float(Sample_y['LKnee_y'])

        PCM['R Thigh_x']=XPR+cm*(XDR-XPR)
        PCM['R Thigh_y']=YPR+cm*(YDR-YPR)
        PCM['L Thigh_x']=XPL+cm*(XDL-XPL)
        PCM['L Thigh_y']=YPL+cm*(YDL-YPL)

        ############################# Shank #########################################
        if Gender == 'Male':
            cm=Male_Segment_Length_Percentage['Shank']
        else: cm=Female_Segment_Length_Percentage['Shank']
            
        XPR=float(Sample_x['RKnee_x'])
        YPR=float(Sample_y['RKnee_y'])
        XDR=float(Sample_x['RAnkle_x'])
        YDR=float(Sample_y['RAnkle_y'])

        XPL=float(Sample_x['LKnee_x'])
        YPL=float(Sample_y['LKnee_y'])
        XDL=float(Sample_x['LAnkle_x'])
        YDL=float(Sample_y['LAnkle_y'])

        PCM['R Shank_x']=XPR+cm*(XDR-XPR)
        PCM['R Shank_y']=YPR+cm*(YDR-YPR)
        PCM['L Shank_x']=XPL+cm*(XDL-XPL)
        PCM['L Shank_y']=YPL+cm*(YDL-YPL)

        ############################ Foot ##############################################
        PCM['R Foot_x']=Sample_x['RAnkle_x']
        PCM['R Foot_y']=Sample_y['RAnkle_y']
        PCM['L Foot_x']=Sample_x['LAnkle_x']
        PCM['L Foot_y']=Sample_y['LAnkle_y']
        
        Frame[str(i)]=PCM
    
    return Frame
    
# TCM calculation
def Calculate_TCM(PCM_Frames, m=80):
    
    '''
    Computes the TCM given PCM values for all frames, simply looping over every frame and multiplying
    by its mass from the mass dictionary
    
    '''

    
    num_frames=len(PCM_Frames)
    keys=list(PCM_Frames['0'].keys())

    Mass = m # This is just an arbitrary value
    Mass_dict= Get_Mass(Mass)
    
    TCM_x=[]
    TCM_y=[]

    for i in range(num_frames):
        x=[]
        y=[]
        x.append(float(PCM_Frames[str(i)]['Head & Neck_x'])*Mass_dict['Head & Neck'])
        y.append(float(PCM_Frames[str(i)]['Head & Neck_y'])*Mass_dict['Head & Neck'])
        x.append(float(PCM_Frames[str(i)]['Trunk_x'])*Mass_dict['Trunk'])
        y.append(float(PCM_Frames[str(i)]['Trunk_y'])*Mass_dict['Trunk'])
        x.append(float(PCM_Frames[str(i)]['R Upper Arm_x'])*Mass_dict['Upper Arm'])
        y.append(float(PCM_Frames[str(i)]['R Upper Arm_y'])*Mass_dict['Upper Arm'])
        x.append(float(PCM_Frames[str(i)]['L Upper Arm_x'])*Mass_dict['Upper Arm'])
        y.append(float(PCM_Frames[str(i)]['L Upper Arm_y'])*Mass_dict['Upper Arm'])
        x.append(float(PCM_Frames[str(i)]['R Forearm_x'])*Mass_dict['Forearm'])
        y.append(float(PCM_Frames[str(i)]['R Forearm_y'])*Mass_dict['Forearm'])
        x.append(float(PCM_Frames[str(i)]['L Forearm_x'])*Mass_dict['Forearm'])
        y.append(float(PCM_Frames[str(i)]['L Forearm_y'])*Mass_dict['Forearm'])         
        x.append(float(PCM_Frames[str(i)]['R Hand_x'])*Mass_dict['Hand'])
        y.append(float(PCM_Frames[str(i)]['R Hand_y'])*Mass_dict['Hand'])
        x.append(float(PCM_Frames[str(i)]['L Hand_x'])*Mass_dict['Hand'])
        y.append(float(PCM_Frames[str(i)]['L Hand_y'])*Mass_dict['Hand'])
        x.append(float(PCM_Frames[str(i)]['R Thigh_x'])*Mass_dict['Thigh'])
        y.append(float(PCM_Frames[str(i)]['R Thigh_y'])*Mass_dict['Thigh'])
        x.append(float(PCM_Frames[str(i)]['L Thigh_x'])*Mass_dict['Thigh'])
        y.append(float(PCM_Frames[str(i)]['L Thigh_y'])*Mass_dict['Thigh'])
        x.append(float(PCM_Frames[str(i)]['R Shank_x'])*Mass_dict['Shank'])
        y.append(float(PCM_Frames[str(i)]['R Shank_y'])*Mass_dict['Shank'])
        x.append(float(PCM_Frames[str(i)]['L Shank_x'])*Mass_dict['Shank'])
        y.append(float(PCM_Frames[str(i)]['L Shank_y'])*Mass_dict['Shank'])
        x.append(float(PCM_Frames[str(i)]['R Foot_x'])*Mass_dict['Foot'])
        y.append(float(PCM_Frames[str(i)]['R Foot_y'])*Mass_dict['Foot'])
        x.append(float(PCM_Frames[str(i)]['L Foot_x'])*Mass_dict['Foot'])
        y.append(float(PCM_Frames[str(i)]['L Foot_y'])*Mass_dict['Foot'])

       
        TCM_x.append(sum(x)/Mass)
        TCM_y.append(sum(y)/Mass)      

    return np.array(TCM_x), np.array(TCM_y)

def Calculate_L(TCM_x, TCM_y, PCM_Frames):
    '''
    Computes the length of the segment (part) simply by sqrt((x2 - x1)^2 + (y2 - y1)^2)
    '''
    
    L=[]

    for i in range(len(PCM_Frames)):
        Values=list(PCM_Frames[str(i)].values())
        x=[(np.square(float(TCM_x[i])- float(Values[j]))) for j in range(len(Values)) if j%2==0]
        y=[(np.square(float(TCM_y[i])- float(Values[j]))) for j in range(len(Values)) if j%2==1]
        
        L.append([np.sqrt(float(k)+float(l)) for k,l in zip(x,y)])
        
    return np.array(L)

def Calculate_D(PCM_Frames, TCM_x, TCM_y , Type='Radians'):
    '''
    C   -> TCM
    C6  -> Left Upper Arm
    C7  -> Left Forearm
    C12 -> Right foot
    C15 -> Left foot
    '''
    
    D1=[]
    D2=[]
    D3=[]
    for i in range(len(PCM_Frames)):  # Gathering the x, y coords together to compute the vectors in a convenient way

        C   = np.array([TCM_x[i], TCM_y[i]])
        C6  = np.array([PCM_Frames[str(i)]['L Upper Arm_x'], PCM_Frames[str(i)]['L Upper Arm_y']], dtype=np.float)
        C7  = np.array([PCM_Frames[str(i)]['L Forearm_x'], PCM_Frames[str(i)]['L Forearm_y']],dtype=np.float)
        C12 = np.array([PCM_Frames[str(i)]['R Foot_x'], PCM_Frames[str(i)]['R Foot_y']],dtype=np.float)
        C15 = np.array([PCM_Frames[str(i)]['L Foot_x'], PCM_Frames[str(i)]['L Foot_y']],dtype=np.float)

        C6C = C - C6          # Computing the vectors
        C6C7 = C7 - C6
        CC12 = C12 - C
        CC15 = C15 - C
        C12C15 = C15 - C12

        C6C_mag = np.linalg.norm(C6C)    # Computing the magnitude or norm or ||vector||
        C6C7_mag = np.linalg.norm(C6C7)
        CC12_mag = np.linalg.norm(CC12)
        CC15_mag = np.linalg.norm(CC15)
        C12C15_mag = np.linalg.norm(C12C15)

        #D1.append(np.arccos(np.linalg.norm((CC12-CC15))/(CC12_mag*CC15_mag))) #this is the paper's algorithm
        #D2.append(np.arccos(np.linalg.norm((C6C-C6C7))/(C6C_mag*C6C7_mag)))   #this is the paper's algorithm
        D3.append(np.linalg.norm((CC12*C12C15))/(C12C15_mag)) 
        
        D1.append(np.arccos(np.dot(CC12, CC15)/(CC12_mag*CC15_mag))) # My implimentation for the angle
        D2.append(np.arccos(np.dot(C6C, C6C7)/(C6C_mag*CC15_mag)))   # My implimentation for the angle

    if Type == 'Radians': return np.array(D1), np.array(D2), np.array(D3)
    if Type == 'Degrees': return np.array(D1)*180/np.pi, np.array(D2)*180/np.pi, np.array(D3)
    
def Calculate_R(PCM_Frames):
    '''
    Looping over every frame and subtracting from the previous frame
    Computes the change of PCM coords
    '''
    
    R=[]

    values_c=list(PCM_Frames['0'].values())
    x_c=[values_c[i] for i in range(len(values_c)) if i%2==0]    # As before gathering x, y together to make a vector
    y_c=[values_c[i] for i in range(len(values_c)) if i%2==1]
    vector_c=np.array([[q,w] for q,w in zip(x_c,y_c)])
    zero=np.zeros_like(np.linalg.norm(vector_c, axis=1))
    R.append(zero) # Appending 0 for the first frame because it has no previous frame, so the change is zero

    for i in range(1,len(PCM_Frames)): # Subtracting the previous frame from the current frame which are represented in vectors of x, y coords

        values_c=list(PCM_Frames[str(i-1)].values())
        x_c=[values_c[k] for k in range(len(values_c)) if k%2==0]
        y_c=[values_c[k] for k in range(len(values_c)) if k%2==1]
        vector_c=np.array([[q,w] for q,w in zip(x_c,y_c)], dtype= np.float)

        values_next=list(PCM_Frames[str(i)].values())
        x_next=[values_next[k] for k in range(len(values_next)) if k%2==0]
        y_next=[values_next[k] for k in range(len(values_next)) if k%2==1]
        vector_next=np.array([[q,w] for q,w in zip(x_next, y_next)], dtype= np.float)
        R.append(list(np.linalg.norm(vector_next - vector_c, axis=1)))
    return np.array(R)

def Add_Features_To_dataframe(File, PCM_Frames, TCM_x, TCM_y, L, R, D1, D2, D3):
    
    keys=list(PCM_Frames['0'].keys())

    ################################# Creates columns ######################################
    for i in keys:
        File[i+'_PCM']=0    # Creates the columns
    
    File['TCM_x']=TCM_x      # Creates the columns and adds the value
    File['TCM_y']=TCM_y
    
    File['Head & Neck_L']=L[:,0]
    File['Trunk_L']=L[:,1]
    
    for i in range(4,len(keys),2):      # Removes the _x to put the name correctly without coords
        File[keys[i].replace('_x','')+'_L']=0
        
    
    File['Head & Neck_R']=R[:,0]
    File['Trunk_R']=R[:,1]
    
    for i in range(4,len(keys),2):
        File[keys[i].replace('_x','')+'_R']=0
        
    File['D1']= D1
    File['D2']= D2
    File['D3']= D3

################################ Add values ###########################################

    for i in range(len(PCM_Frames)):
        File.iloc[i,37:65]=list(PCM_Frames[str(i)].values())
    
    for i in range(len(L)):
        File.iloc[i,69:81]=L[i,2:]
    
    for i in range(len(R)):
        File.iloc[i,83:95]=R[i,2:] 
    
    return File

def test_L_Feature(L, Frame_num):         # Just to test the L feature, just ignore :)

    '''
    They are ordered as follows :

    Head & Neck     
    Trunk
    R Upper Arm
    L Upper Arm
    R Forearm
    L Forearm
    R Hand
    L Hand
    R Thigh
    L Thigh
    R Shank
    L Shank
    R Foot
    L Foot

    '''
    print('Shape of L = ', L.shape)
    print('L = ',L[Frame_num])
    print('')
    print('TCM_x of frame {} = {}'.format(Frame_num,TCM_x[Frame_num]))
    print('TCM_y of frame {} = {}'.format(Frame_num,TCM_y[Frame_num]))
    print('')
    print('  \t| PCM of frame ', Frame_num, ' |')
    print(PCM_Frames[str(Frame_num)])


# In[24]:

'''
File=pd.read_csv('NEW_TEST.csv').iloc[:,1:]
File.head()


# In[25]:


# Extracting Coordinates
X_Coords, Y_Coords= Get_Coords(File)


# In[26]:


print('X_Coords shape = ', X_Coords.shape)
X_Coords.head()


# In[27]:


print('Y_Coords shape = ', Y_Coords.shape)
Y_Coords.head()


# In[28]:


import time

start=time.time()
PCM_Frames= Calculate_PCM(X_Coords, Y_Coords)
print('Time taken for PCM for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))

start=time.time()
TCM_x, TCM_y= Calculate_TCM(PCM_Frames)
print('Time taken for TCM for {} Frames is {:.3f} ms'.format(len(TCM_x),(time.time()-start)*1000))
print('')
print('TCM_x = ',TCM_x)
print('TCM_y = ',TCM_y)


# In[29]:


start=time.time()
L= Calculate_L(TCM_x, TCM_y, PCM_Frames)
print('Time taken for L features for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))


# In[30]:


Frame_num=40
test_L_Feature(L, Frame_num)


# In[31]:


start=time.time()
D1, D2, D3 = Calculate_D(PCM_Frames, TCM_x, TCM_y, 'Degrees')
print('Time taken for D1, D2, D3 features for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
print('')
print(D1)
print('')
print(D2)
print('')
print(D3)


# In[11]:


start=time.time()
R=Calculate_R(PCM_Frames)
print('Time taken for R feature for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
print('Shape of R = ',R.shape)


# In[12]:


R

'''
# In[135]:

'''
import time
start=time.time()

File=pd.read_csv('NEW_TEST.csv').iloc[:,1:]
X_Coords, Y_Coords= Get_Coords(File)
PCM_Frames= Calculate_PCM(X_Coords, Y_Coords)
TCM_x, TCM_y= Calculate_TCM(PCM_Frames)
L= Calculate_L(TCM_x, TCM_y, PCM_Frames)
D1, D2, D3 = Calculate_D(PCM_Frames, TCM_x, TCM_y, 'Degrees')
R=Calculate_R(PCM_Frames)
out=Add_Features_To_dataframe(File, PCM_Frames, TCM_x, TCM_y, L, R, D1, D2, D3)

print('Time taken the for whole file of {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))

out.to_csv("Final_Features.csv")
print('Success!')

out.head()

'''