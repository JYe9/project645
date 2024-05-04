import read_bvh
import numpy as np
from os import listdir
import os
import transforms3d.euler as euler

def euler_to_rotation_matrix(euler_angles):
    # Convert euler angles to rotation matrix
    R_x = euler.euler2mat(euler_angles[0], 0, 0)
    R_y = euler.euler2mat(0, euler_angles[1], 0)
    R_z = euler.euler2mat(0, 0, euler_angles[2])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def rotation_matrix_to_6D(R):
    # Convert rotation matrix to 6D representation
    return R[:2, :].flatten()

def _6D_to_rotation_matrix(rep_6D):
    # Convert 6D representation to rotation matrix
    R = np.zeros((3, 3))
    R[:2, :] = rep_6D.reshape(2, 3)
    R[2, :] = np.cross(R[0, :], R[1, :])
    return R

def generate_6D_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    # TODO:
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names = listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len = len(bvh_dance_name)
        if name_len > 4:
            if bvh_dance_name[name_len - 4:] == ".bvh":
                bvh_data = read_bvh.parse_frames(src_bvh_folder + bvh_dance_name)
                euler_data = bvh_data[:, 3:]
                rep_6D_data = []
                for frame in euler_data:
                    euler_angles = frame.reshape(-1, 3)
                    rep_6D_frame = []
                    for joint_angles in euler_angles:
                        R = euler_to_rotation_matrix(joint_angles)
                        rep_6D = rotation_matrix_to_6D(R)
                        rep_6D_frame.append(rep_6D)
                    rep_6D_data.append(np.concatenate(rep_6D_frame))
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", np.array(rep_6D_data))

def generate_bvh_from_6D_traindata(src_train_folder, tar_bvh_folder):
    # TODO:
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
    dances_names = listdir(src_train_folder)
    for dance_name in dances_names:
        name_len = len(dance_name)
        if name_len > 4:
            if dance_name[name_len - 4:] == ".npy":
                rep_6D_data = np.load(src_train_folder + dance_name)
                num_frames = rep_6D_data.shape[0]
                num_joints = int(rep_6D_data.shape[1] / 6)
                bvh_data = np.zeros((num_frames, num_joints * 3 + 6))
                for i in range(num_frames):
                    rep_6D_frame = rep_6D_data[i]
                    euler_frame = []
                    for j in range(num_joints):
                        rep_6D = rep_6D_frame[j * 6:(j + 1) * 6]
                        R = _6D_to_rotation_matrix(rep_6D)
                        euler_angles = euler.mat2euler(R)
                        euler_frame.append(euler_angles)
                    bvh_data[i, 6:] = np.concatenate(euler_frame)
                read_bvh.write_frames(standard_bvh_file, tar_bvh_folder + dance_name + ".bvh", bvh_data)

standard_bvh_file = "../train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

# Encode data from bvh to positional encoding
# generate_6D_traindata_from_bvh("../train_data_bvh/martial/","../train_data_6D/martial/")
# generate_6D_traindata_from_bvh("../train_data_bvh/indian/","../train_data_6D/indian/")
generate_6D_traindata_from_bvh("../train_data_bvh/salsa/","../train_data_6D/salsa/")

# Decode from positional to bvh
# generate_bvh_from_6D_traindata("../train_data_6D/martial/", "../test_data_6D_bvh/martial/",)
# generate_bvh_from_6D_traindata("../train_data_6D/indian/", "../test_data_6D_bvh/indian/",)
generate_bvh_from_6D_traindata("../train_data_6D/salsa/", "../test_data_6D_bvh/salsa/",)
