# Tutorial for the dense gt generation tools
## Categorical mapping and clustering for the semantic segmentation annotations.
*change the source file path and save file path according to the comments.
*python lidarseg_annotoos.py to convert semantic segmentation labels.
*Note that currently the data are just label, containing no raw lidar point data.
## Concate annotation point-wise to the raw point dataset
*change the source file path and save file path according to the comments.
*changed the processed annotation path according to the comment.
*changes nusc info path while loading nusc dataset class.
*python lidar_seg_label_cating.py
## Dense lidar point cloud aggregation
*change the source file path and save file path according to the comments.
*changes nusc info path while loading nusc dataset class.
*python ego.py
*Note that this step need large storation space
## Image-wise annotation generation
*change the source file path and save file path in the label_generator function. (save_path indicates the storation path for the image-wise top view label and the source path
*is the aggregated point cloud path)
*uncomment #label_generator() and comment colorized_image_generator() to generate label leveraged for training and testing without colorization
*python gt_img.py
*(voxellization need to be compiled before execution)
