import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
import pandas as pd
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    out = manip(humans, directory='/home/minamelek/Documents/Work/BrainWise/O-Nect-master/Humans.csv')
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()
