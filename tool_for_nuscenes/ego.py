#%matplotlib inline
from nuscenes import NuScenes
import os
import numpy as np
import torch
import json
import sys
import glob
import logging
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
logging.basicConfig(level=logging.DEBUG)
file_path = "/mrtstorage/users/kpeng/nu_lidar_seg/concat_lidar_flat_divided_new/" #path to the point cloud concatenated with semantic labels
save_path = "/mrtstorage/users/kpeng/nu_lidar_seg/concat_lidar_flat_divided/new_2/" #save path for dense aggregated labeled point cloud (note that this step will take a large space)
def seg_concat():
    nusc = NuScenes(version='v1.0-trainval', dataroot='/path/to/nuscenes/v1.0-trainval/', verbose=True) #path to v1.0-trianval for the raw nuscenes dataset 
    scene_list = {}
    scene_token = {}
    scene_ego = {}
    count = 0
    for scene in nusc.scene:

        prev_token = scene["first_sample_token"]
        last_token = scene["last_sample_token"]
        prev_sample_token = nusc.get('sample',prev_token)['data']['LIDAR_TOP']
        prev_filename = nusc.get('sample_data', prev_sample_token)['filename']
        scene_list[str(count)]=[]
        scene_token[str(count)]=[]
        scene_ego[str(count)]=[]
        if prev_filename.split('/')[0] == 'samples':
            scene_list[str(count)].append(prev_filename)
            scene_token[str(count)].append(prev_sample_token)
            scene_ego_token = nusc.get('sample_data', prev_sample_token)['ego_pose_token']
            scene_ego[str(count)].append(nusc.get('ego_pose', scene_ego_token))
        count_n = 0
        while True:
            next_token = nusc.get('sample_data', prev_sample_token)['next']
            if next_token == "":
                break
            next_filename = nusc.get('sample_data', next_token)['filename']
            next_ego_token = nusc.get('sample_data', next_token)['ego_pose_token']
            if next_filename.split('/')[0] == 'samples':
                scene_ego[str(count)].append(nusc.get('ego_pose', next_ego_token))
                scene_list[str(count)].append(next_filename)
                scene_token[str(count)].append(next_token)
                count_n += 1
            prev_sample_token = next_token
            if count_n == scene["nbr_samples"]-1:
                break
        count +=1
    return scene_list, scene_token, scene_ego,nusc,count
def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def pc_ego_list(scene_list_1,scene_token_1,scene_ego_1,nusc, count):
        for scene_idx in range(len(scene_list_1.keys())):
            key = str(scene_idx)
            scene_file_list = scene_list_1[key]
            scene_token_list = scene_token_1[key]
            scene_ego_list = scene_ego_1[key]
            num_samples = len(scene_token_list)
            for idx in range(len(scene_token_list)):
                scene_token = scene_token_list[idx]
                key_frame_sample = nusc.get('sample_data', scene_token)['filename']
                key_frame = nusc.get('sample_data', scene_token)
                calibrated_sensor_token = key_frame['calibrated_sensor_token']
                key_rotation = nusc.get('calibrated_sensor', calibrated_sensor_token)['rotation']
                key_trans = nusc.get('calibrated_sensor', calibrated_sensor_token)['translation']

                k_r_e = transform_matrix(key_trans, Quaternion(key_rotation),True)

                key_pc =  np.fromfile(file_path+key_frame_sample, dtype=np.float32, count=-1).reshape(-1,6)
                scene_ego_idx = scene_ego_list[idx]
                scene_ego_trans = np.array(scene_ego_idx["translation"])
                scene_ego_rot = np.array(scene_ego_idx["rotation"]).tolist()
                scene_ego_timestamp = scene_ego_idx["timestamp"]
                threshold = 2 * np.max(np.linalg.norm(key_pc[:,:3]))
                transform_matrix_k= transform_matrix(scene_ego_trans, Quaternion(scene_ego_rot),True)

                final_pc = key_pc
                for id in range(num_samples):
                    scene_token_re = scene_token_list[id]
                    scene_re_idx = scene_ego_list[id]
                    translation_re = np.array(scene_re_idx["translation"])
                    rot_re = np.array(scene_re_idx["rotation"]).tolist()
                    r_sensor_token = nusc.get('sample_data', scene_token)['calibrated_sensor_token']
                    rot_to_ego = nusc.get('calibrated_sensor', r_sensor_token)['rotation']
                    trans_to_ego = nusc.get('calibrated_sensor', r_sensor_token)['translation']

                    r_r_e = transform_matrix(trans_to_ego, Quaternion(rot_to_ego))
                    transform_matrix_r= transform_matrix(translation_re, Quaternion(rot_re))
                    distance = np.linalg.norm(scene_ego_trans - translation_re)
                    if distance <= threshold:
                        sample_re = nusc.get('sample_data', scene_token_re)['filename']
                        re_pc =  np.fromfile(file_path+sample_re, dtype=np.float32, count=-1).reshape(-1,6)
                        anno_seg_re = re_pc[:,-1]
                        mask_flat =(anno_seg_re ==1)|(anno_seg_re==8)| (anno_seg_re == 11) | (anno_seg_re == 12) | (anno_seg_re == 13) | (anno_seg_re == 14) | (anno_seg_re == 15) | (anno_seg_re == 16)
                        re_pc_flat = re_pc[mask_flat] # point_num, [x,y,z,r,t,seg_anno]
                        p_n = re_pc_flat.shape[0]
                        homo = np.concatenate((re_pc_flat[:,:3],np.ones((p_n,1))),axis=-1)
                        re_pc_flat[:,:3] = (((k_r_e@(transform_matrix_k @ (transform_matrix_r @ (r_r_e@ homo.T))))).T)[:,:3]
                        final_pc = np.concatenate((final_pc,re_pc_flat),axis=0)

                binfile = open(save_path+key_frame_sample, 'wb+')
                binfile.write(final_pc.flatten().tobytes())
                binfile.close()
                print(save_path+key_frame_sample)
                sys.exit()

if __name__ == "__main__":
    a,b,c,d,e =seg_concat()

