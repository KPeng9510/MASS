#CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --extra_tag 'pillar_obs_dense_att_simunet_wda'
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt '/home/ki/hdd0/output/juncong/semantickitti/output/nuscenes_models/cbgs_pp_multihead/pillar_obs_dense_att_simunet_wda/ckpt/checkpoint_epoch_17.pth' --extra_tag 'pillar_obs_dense_att_simunet_wda'
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --extra_tag 'pillar_obs_dense_att_simunet_wda' --eval_all

# pillar_obs_dense_concat_simunet_wda
#CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --extra_tag 'pillar_obs_dense_concat_simunet_wda1'


#CUDA_VISIBLE_DEVICES=0 python test.py --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --ckpt '/home/ki/hdd0/output/juncong/semantickitti/output/nuscenes_models/cbgs_pp_multihead/pillar_obs_dense_concat_simunet_wda/ckpt/checkpoint_epoch_28.pth' --extra_tag 'pillar_obs_dense_concat_simunet_wda'
