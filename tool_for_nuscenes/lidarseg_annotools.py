import sys
import glob
import os
import json
import numpy as np
import torch
filepath="/cvhci/data/nuscenes/lidarseg/v1.0-trainval" #path to semantic segmentation labels for nuscenes lidarseg dataset
savepath="/cvhci/temp/kpeng/processed_anno_new/" # path to the saved processed labels
"""
'car:5',    1
'truck:5', 2
 'construction_vehicle:5',5
 'bus:5', 3,4
 'trailer:5', 9

'barrier:5',21
 'motorcycle:5', 6
'bicycle:5',7
 'pedestrian:5',12,13,14,18
 'traffic_cone:5',20
flat

"""
def eachFile(filepath):
    pathDir =  os.listdir(filepath)

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    dic = {}
    for file_name in pathDir:
        file_path = os.path.join('%s/%s' % (filepath, file_name))
        save_file_path = os.path.join('%s/%s' % (savepath, file_name))
        semantic_labels = torch.from_numpy(np.fromfile(file_path, dtype=np.uint8,count=-1))
        #print(torch.max(semantic_labels))
        points_number = len(semantic_labels)
        new_label = torch.zeros([points_number], dtype=semantic_labels.dtype, device=semantic_labels.device)
        
        
        mask_barrier = semantic_labels == 9
        mask_bicycle= semantic_labels == 14
        mask_bus = (semantic_labels == 15)|(semantic_labels == 16)
        mask_car = semantic_labels == 17
        mask_c_v = semantic_labels == 18
        mask_motocycle = semantic_labels == 21
        mask_ped = (semantic_labels == 2) | (semantic_labels == 3) | (semantic_labels == 4) | (semantic_labels==5) | (semantic_labels==6) | (semantic_labels==7)|(semantic_labels==8)
        mask_cone = semantic_labels == 12
        mask_trailer = semantic_labels == 22
        mask_truck = semantic_labels == 23
        mask_drive = semantic_labels == 24
        mask_other_flat = semantic_labels == 25
        mask_sidewalk = semantic_labels == 26
        mask_terrain = semantic_labels == 27
        mask_manmade = semantic_labels == 28
        mask_vegetation = semantic_labels == 30


        new_label[mask_barrier] = 1
        new_label[mask_bicycle] = 2
        new_label[mask_bus] = 3
        new_label[mask_car] = 4
        new_label[mask_c_v] = 5
        new_label[mask_motocycle] = 6
        new_label[mask_ped] = 7
        new_label[mask_cone] = 8
        new_label[mask_trailer] = 9
        new_label[mask_truck] =10
        new_label[mask_drive] = 11
        new_label[mask_other_flat] = 12
        new_label[mask_sidewalk] = 13
        new_label[mask_terrain] = 14
        new_label[mask_manmade] = 15
        new_label[mask_vegetation] = 16

        binfile = open(save_file_path, 'wb+') 
        binfile.write(new_label.numpy().tobytes())


if __name__ == "__main__":
    eachFile(filepath)

