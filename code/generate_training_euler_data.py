import read_bvh
import numpy as np
from os import listdir
import os


def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    # TODO:
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names = listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len = len(bvh_dance_name)
        if name_len > 4:
            if bvh_dance_name[name_len - 4:] == ".bvh":
                bvh_data = read_bvh.parse_frames(src_bvh_folder + bvh_dance_name)
                euler_data = bvh_data[:, 3:]  # Assuming the first 3 columns are translation data
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", euler_data)

def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    # TODO:
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
    dances_names = listdir(src_train_folder)
    for dance_name in dances_names:
        name_len = len(dance_name)
        if name_len > 4:
            if dance_name[name_len - 4:] == ".npy":
                euler_data = np.load(src_train_folder + dance_name)
                num_frames = euler_data.shape[0]
                bvh_data = np.zeros((num_frames, euler_data.shape[1] + 3))
                bvh_data[:, 3:] = euler_data
                read_bvh.write_frames("../train_data_bvh/standard.bvh", tar_bvh_folder + dance_name + ".bvh", bvh_data)


standard_bvh_file = "../train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

print('skeleton: ', skeleton)

# Encode data from bvh to positional encoding
# generate_euler_traindata_from_bvh("../train_data_bvh/martial/","../train_data_euler/martial/")
# generate_euler_traindata_from_bvh("../train_data_bvh/indian/","../train_data_euler/indian/")
generate_euler_traindata_from_bvh("../train_data_bvh/salsa/","../train_data_euler/salsa/")
# Decode from positional to bvh
# generate_bvh_from_euler_traindata("../train_data_euler/martial/", "../test_data_euler_bvh/martial/")
# generate_bvh_from_euler_traindata("../train_data_euler/indian/", "../test_data_euler_bvh/indian/")
generate_bvh_from_euler_traindata("../train_data_euler/salsa/", "../test_data_euler_bvh/salsa/")
