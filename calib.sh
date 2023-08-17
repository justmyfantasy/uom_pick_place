#!/bin/bash
DATA_ROOT=0803
CAP_DATA_DIR=${DATA_ROOT}/capture_1
CAP_DET_DIR=${CAP_DATA_DIR}_det
ROBOT_POSE_DIR=${DATA_ROOT}/robot_pose_data

python calib_xyz/aruco_det.py \
    --data_dir $CAP_DATA_DIR \
    --save_dir ${CAP_DET_DIR}

python calib_xyz/hand_eye_calib_xyz.py \
    --rmat_dir ${ROBOT_POSE_DIR}/R_matrix.csv \
    --tmat_dir ${ROBOT_POSE_DIR}/T_matrix.csv \
    --det_res_dir ${CAP_DET_DIR} \
    --calib_save_dir ${DATA_ROOT} \
    --z_source aruco_pose  # "aruco_pose" or "cam_depth"