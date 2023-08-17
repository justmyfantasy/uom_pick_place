import argparse
import os
import cv2
from PIL import Image
import numpy as np
import pickle
from thread_kinect2 import FN2Feed

from yolo.yolo import YOLO

yolo = YOLO()

video_getter = FN2Feed().start()

intrins = np.array([[1081.3720703125, 0, 959.5],
                [0, 1081.3720703125, 539.5],
                [0, 0, 1]])
distort = np.array([[0.01502064, 0.05236846,  0.00296999,  0.00318522, -0.11516424]])

with open('calib_trans.pkl', 'rb') as f:
    calib_trans = pickle.load(f)
image_to_arm = calib_trans['image2arm']

def convert_color(rgb):
    i_image = np.uint8(rgb)
    i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
    i_image = Image.fromarray(np.uint8(i_image))
    return i_image

def bbox2xyz(bbox, depth):
    xyz = np.zeros(3)
    center = np.array(bbox).reshape(2,2).mean(axis=0).astype(np.uint)
    xyz[2] = depth[center[0], center[1]] * 1e-3
    
    xyz[0] = (center[1] - intrins[0, 2]) / intrins[0, 0] * xyz[2] 
    xyz[1] = - (1080 - center[0] - intrins[1, 2]) / intrins[1, 1] * xyz[2] # y axis is opposite for aruco-estimated pose and kinectv2 pose

    return xyz

while True:
    if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
        video_getter.stop()
        break

    color = video_getter.color
    depth = video_getter.bigdepth
    if color is not None and depth is not None:
        i_image = convert_color(color)
        r_image, det_res = yolo.detect_image(i_image, ret_box=True)
        r_image = np.array(r_image)

        depth[np.isinf(depth)] = 0
        depth = depth[1:-1]
        # extract camera coords and transform to robot coords
        for det in det_res:
            cam_xyz = bbox2xyz(det['bbox'], depth)
            # if cam_xyz[2] < 0.4: 
            #     print('object center depth unreliable.')
            #     continue
            det['cam_xyz'] = cam_xyz
            det['robot_xyz'] = image_to_arm @ np.insert(cam_xyz, -1, 1)
            cv2.putText(r_image, '(%.02f,%.02f,%.02f)'%(cam_xyz[0], cam_xyz[1], cam_xyz[2]), 
                        (det['bbox'][1], det['bbox'][0]+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
            cv2.putText(r_image, '(%.02f,%.02f,%.02f)'%(det['robot_xyz'][0], det['robot_xyz'][1], det['robot_xyz'][2]), 
                        (det['bbox'][1], det['bbox'][0]+45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 

        # convert back for show
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', r_image)

        
