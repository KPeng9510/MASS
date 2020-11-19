from .detector3d_template import Detector3DTemplate
from .segmentation_head import FCNMaskHead
import sys
from .erfnet import Net
import torch.nn.functional as F
class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.segmentation_head = FCNMaskHead()
    def forward(self, batch_dict):
        module_index = 0
        for cur_module in self.module_list:
            module_index += 1
            batch_dict = cur_module(batch_dict)
            if module_index == 3:
                #print(batch_dict.keys())
                #print(batch_dict["spatial_features_2d"].size())
                #print(batch_dict["spatial_features"].size())
                spatial_features = batch_dict["spatial_features_2d"]
                pred_seg = self.segmentation_head(spatial_features)
                targets = batch_dict['one_hot']
                #print(pred_seg.size())
                #print(targets.size())
                loss_seg = F.binary_cross_entropy_with_logits(pred_seg,targets,reduction='mean')
                #print(loss_seg)
                #sys.exit()
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss+loss_seg
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
