import os, sys
import os.path as osp
import numpy as np
import argparse
import cv2
from glob import glob
import pickle
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True, help="path to data")
ap.add_argument("-s", "--save_dir", required=True, help="path to save")
args = vars(ap.parse_args())

intrins = np.array([[1081.3720703125, 0, 959.5],
                [0, 1081.3720703125, 539.5],
                [0, 0, 1]])
distort = np.array([[0.01502064, 0.05236846,  0.00296999,  0.00318522, -0.11516424]])

data_dir = args['data_dir']
img_list = glob(osp.join(data_dir, 'color*'))
img_list.sort()
# print(img_list)

big_depth_list = glob(osp.join(data_dir, 'big_depth*'))
big_depth_list.sort()
# print(big_depth_list)

save_dir = args['save_dir']
os.makedirs(save_dir, exist_ok=True)

for img_path, big_depth_path in tqdm(zip(img_list, big_depth_list)):
    image = cv2.imread(img_path)
    frame = cv2.flip(image, 1)

    with open(big_depth_path, 'rb') as f:
        big_depth = pickle.load(f)
    big_depth[np.isinf(big_depth)] = 0
    big_depth = big_depth[1:-1]

    save_dict = {}
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_100)
    arucoParams = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    save_dict['corners'] = corners
    save_dict['ids'] = ids

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.095, intrins, distort)
            print(rvec)
            print(cv2.Rodrigues(rvec)[0])
            print(tvec)

            xyz = np.zeros(3)
            center = np.average(corners[0][0], axis=0).astype(np.uint)
            xyz[2] = big_depth[center[1], center[0]] * 1e-3
            
            xyz[0] = (center[0] - intrins[0, 2]) / intrins[0, 0] * xyz[2]
            xyz[1] = (1080 - center[1] - intrins[1, 2]) / intrins[1, 1] *xyz[2]
            
            save_dict['rvec'] = rvec
            save_dict['rmat'] = cv2.Rodrigues(rvec)[0]
            save_dict['tvec'] = tvec
            save_dict['xyz'] = xyz
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            # Draw Axis
            cv2.aruco.drawAxis(frame, intrins, distort, rvec, tvec, 0.1)

    img_name = osp.split(img_path)[1]
    cv2.imwrite(osp.join(save_dir, img_name), frame)
    with open(osp.join(save_dir, osp.splitext(img_name)[0] + '.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)
