import os
import os.path as osp
import numpy as np
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--rmat_dir", required=True, help="path to rmat")
ap.add_argument("--tmat_dir", required=True, help="path to tmat")
ap.add_argument("--det_res_dir", required=True, help="path to aruco marker detect results")
ap.add_argument("--calib_save_dir", default=".", help="path to save calibration results")
ap.add_argument("--z_source", choices=["aruco_pose", "cam_depth"], default="aruco_pose")
args = vars(ap.parse_args())

# load robot pose data
R_csv = np.loadtxt(args["rmat_dir"], delimiter=",", dtype=str)
R_gripper2base = R_csv[1:, 10:].astype(np.float64).reshape((9,3,3))
R_base2gripper = np.array([x.T for x in R_gripper2base])

t_csv = np.loadtxt(args["tmat_dir"], delimiter=",", dtype=str)
t_gripper2base = t_csv[1:, 4:].astype(np.float64).reshape((9,3,1))*0.001
t_base2gripper = np.array([-r.T @ t for r, t in zip(R_gripper2base, t_gripper2base)])

# load aruco detect data
proc_data_dir = args["det_res_dir"]
proc_files = os.listdir(proc_data_dir)
proc_files = [x for x in proc_files if x.endswith(".pkl")]
proc_files.sort()
rmat_list = []
tvec_list = []
xyz_list = []
for proc_file in proc_files[::3]:
    with open(osp.join(proc_data_dir, proc_file), "rb") as f:
        proc_dict = pickle.load(f)
        rmat_list.append(proc_dict["rmat"])
        tvec_list.append(proc_dict["tvec"][0][0][:, None])
        xyz_list.append(proc_dict["xyz"][:, None])

R_target2cam = np.array(rmat_list)
t_target2cam = np.array(tvec_list)
t_xyz = np.array(xyz_list)

if args["z_source"] == "cam_depth":
    invalid_ids = [3, 7, 8] # the invalid ids need to be manually identified for camera depth
    t_gripper2base = np.delete(t_gripper2base, invalid_ids, axis=0)
    t_xyz = np.delete(t_xyz, invalid_ids, axis=0)

# transformation matrix
if args["z_source"] == "cam_depth":
    centers = np.column_stack((t_xyz[:, :, 0], np.ones(t_xyz.shape[0]).T)).T
elif args["z_source"] == "aruco_pose":
    centers = np.column_stack((t_target2cam[:, :, 0], np.ones(t_target2cam.shape[0]).T)).T
arm_cord = np.column_stack((t_gripper2base[:, :, 0], np.ones(t_gripper2base.shape[0]).T)).T

image_to_arm = np.dot(arm_cord, np.linalg.pinv(centers))
arm_to_image = np.linalg.pinv(image_to_arm)

print("Finished")
print("Image to arm transform:\n", image_to_arm)
print("Arm to Image transform:\n", arm_to_image)
print("Sanity Test:")

print("-------------------")
print("Image_to_Arm")
print("-------------------")
error = []
for ind, pt in enumerate(centers.T):
    import pdb;pdb.set_trace()
    print("Expected:", t_gripper2base[ind, :, 0])
    print("Result:", np.dot(image_to_arm, pt)[:3])
    error.append(np.absolute(t_gripper2base[ind, :, 0] - np.dot(image_to_arm, pt)[0:3]))
print("Average Error:", np.array(error).mean())

calib_trans = {
    'image2arm': image_to_arm,
    'arm2image': arm_to_image,
}
save_path = osp.join(args["calib_save_dir"], "calib_trans.pkl")
with open(save_path, "wb") as f:
    pickle.dump(calib_trans, f)
print('calibration results writed to: ', save_path)
