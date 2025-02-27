import os
import torch
import cv2
import numpy as np


from ..EfficientSAM.efficient_sam import efficient_sam
from .utils import clip_to_image, get_gcn_feature
from dataset.train.utils import uniformsample, four_idx, get_img_gt

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

def four_idx_with_ct(img_gt_poly, center):
    # x_min, y_min = np.min(img_gt_poly, axis=0)
    # x_max, y_max = np.max(img_gt_poly, axis=0)
    # center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
    can_gt_polys = img_gt_poly.copy()
    can_gt_polys[:, 0] -= center[0]
    can_gt_polys[:, 1] -= center[1]
    distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6

    can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
    idx_bottom = np.argmax(can_gt_polys[:, 1]) 
    idx_top = np.argmin(can_gt_polys[:, 1]) 
    idx_right = np.argmax(can_gt_polys[:, 0]) 
    idx_left = np.argmin(can_gt_polys[:, 0]) 
    return [idx_bottom, idx_right, idx_top, idx_left]

class Refine(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=4.):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.feature_dim = 64

        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Conv2d(256, self.feature_dim, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))
        
        self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * self.feature_dim), out_features=num_point * 4, bias=False)
        self.trans_fuse = torch.nn.Linear(in_features=num_point * 4, out_features=num_point * 2, bias=True)

    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)
        
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx, ignore=False):
        if ignore or len(init_polys) == 0:
            return init_polys
        h, w = feature.size(2), feature.size(3)
        poly_num = ct_polys.size(0)
    
        feature = self.trans_feature(feature)

        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
        points = torch.cat([ct_polys, init_polys], dim=1)
        
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1) # (batch_ins_num, C, 129) -> (batch_ins_num, C*129)
        coarse_polys = self.global_deform(feature_points, init_polys) # (batch_ins_num, 128, 2)
        return coarse_polys

class Decode(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., min_ct_score=0.05):
        super(Decode, self).__init__()
        self.stride = init_stride
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score # ct confidence
        self.refine = Refine(c_in=c_in, num_point=num_point, stride=coarse_stride)

        self.contour_dim = 128
        sam_checkpoint_path = os.path.join(PROJECT_PATH, "network", "EfficientSAM", "weights", "efficient_sam_vitt.pt")

        # self.efficientsam = efficient_sam.build_efficient_sam(
        #     encoder_patch_embed_dim=384,
        #     encoder_num_heads=6,
        #     checkpoint=self.sam_checkpoint_path,
        #     ).eval()
        self.efficientsam = efficient_sam.build_efficient_sam(
            encoder_patch_embed_dim=192,
            encoder_num_heads=3,
            checkpoint=sam_checkpoint_path,
            ).eval()
        for param in self.efficientsam.parameters():
            param.requires_grad = False
        self.sam_mode = 'area'

    # output = {}
    def train_decode(self, batch_info_dict: dict, output: dict, cnn_feature: torch.Tensor): 
        """_summary_

        Args:
            data_input (dict): _description_
            output (dict): _description_
            output2 (dict): _description_
            cnn_feature (torch.Tensor): (N,C,H,W)
        """ 
        # wh_pred = output['wh']
        ct_01 = batch_info_dict['ct_01'].bool()  # B num

        ct_ind = batch_info_dict['ct_ind'][ct_01]
        ct_img_idx = batch_info_dict['ct_img_idx'][ct_01]
        wh_gt = batch_info_dict['wh'][ct_01] # h, w = y_max - y_min, x_max - x_min
        box_gt = batch_info_dict['bbox'][ct_01] # h, w = y_max - y_min, x_max - x_min
        _, _, height, width = batch_info_dict['ct_hm'].size()
        ct_x, ct_y = ct_ind % width, ct_ind // width # ct (x,y)
        box_gt_sam = box_gt * self.down_sample
        
        ct_x_, ct_y_ = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)
        ct = torch.cat([ct_x_, ct_y_], dim=1)
        
        batch_pre_masks= []
        self.model_device = ct_01.device

        for i in range(ct_01.shape[0]):
            if (ct_img_idx == i).sum() == 0:
                # the i-th img has no bbox
                continue
            sam_in = batch_info_dict['orig_img'][i].flip(dims=[2]).permute(2,0,1).div(255)[None]
            points = box_gt_sam[ct_img_idx == i].reshape(-1,2,2)[None]
            points_label = torch.tensor([[[2, 3]]], dtype=torch.float32, device=self.model_device).repeat(1, points.size()[1], 1)
            predicted_logits, predicted_iou = self.efficientsam(
                sam_in,
                points,
                points_label
            )

            pre_masks = torch.ge(predicted_logits[0, :, :, :, :], 0).type(torch.int8).to(self.model_device)
            if self.sam_mode == 'iou':
                get_index = predicted_iou[0].argmax(dim=1)
            elif self.sam_mode == 'area':
                get_index = pre_masks.sum(dim=[2,3]).argmax(dim=1)
            # get_index = pre_masks.sum(dim=[2,3]).argmax(dim=1)
            pre_masks = pre_masks[range(get_index.size(0)), get_index]
            batch_pre_masks.append(pre_masks)
        if batch_pre_masks:
            pre_masks = torch.cat(batch_pre_masks, dim=0)
            polys = (self.masks2points_with_ct(pre_masks, ct * self.down_sample) / self.down_sample).to(self.model_device)
        else:
            polys = torch.zeros((0,128), device=self.model_device)

        #init_polys
        output.update({'init_poly':polys})
        # coarsepolys
        coarse_polys = self.refine(cnn_feature, ct, polys, ct_img_idx.clone())
        
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        return
    
    # min_ct_score ==> ct confidence
    def test_decode(self, batch_info_dict, cnn_feature, output, ignore_gloabal_deform=False):
        ct = output['ct']
        detection = output['detection']
        # detection[..., :4] *= self.down_sample
        img_id = torch.zeros((len(ct), ), dtype=torch.int64)

        #####sam#####
        if len(detection) == 0:
            polys = torch.zeros((0,128), device=detection.device)
        else:
            sam_in = batch_info_dict['orig_img'][0].flip(dims=[2]).permute(2,0,1).div(255)[None]
            # sam_in = batch_info_dict['orig_img'][0].to(self.sam_device)[None]
            points = detection[..., :4].reshape(-1,2,2)[None]
            points_label = torch.tensor([[[2, 3]]],dtype=torch.float32, device=sam_in.device).repeat(1, points.size()[1], 1)
             
            predicted_logits, predicted_iou = self.efficientsam(
                sam_in,
                points,
                points_label
            )
            pre_masks = torch.ge(predicted_logits[0, :, :, :, :], 0).type(torch.int8)
            if self.sam_mode == 'iou':
                get_index = predicted_iou[0].argmax(dim=1)
            elif self.sam_mode == 'area':
                get_index = pre_masks.sum(dim=[2,3]).argmax(dim=1)
            pre_masks = pre_masks[range(get_index.size(0)), get_index]

            polys = self.masks2points_with_ct(pre_masks, ct)
        # polys = self.masks2points(pre_masks)# (num_ins, 128)
        # polys = polys / self.down_sample
        output.update({'init_poly':polys})

        poly_coarse = self.refine(cnn_feature, ct / self.down_sample, polys / self.down_sample, img_id, ignore=ignore_gloabal_deform)

        coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        return

    def forward(self, batch_info_dict: dict, cnn_feature:torch.Tensor, output:dict=None, is_training: bool=True, ignore_gloabal_deform=False):
        if is_training:
            self.train_decode(batch_info_dict, output, cnn_feature)
        else:
            self.test_decode(batch_info_dict, cnn_feature, output, ignore_gloabal_deform=ignore_gloabal_deform)
    
    def masks2points(self, instance_masks: torch.Tensor) -> torch.Tensor:
        polys = []
        for mask in instance_masks:
            img = mask.type(torch.uint8).cpu().numpy()
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                ares = []
                for contour in contours:
                    ares.append(cv2.contourArea(contour))
                index = torch.tensor(ares, dtype=torch.float32).argmax()
                contour = contours[index].reshape(-1,2) # n,x,y
            elif len(contours) == 1:
                contour = contours[0].reshape(-1,2) # n,x,y
            else:
                # print(F"There is a non mask in sam out, sum={mask.sum()}.")
                contour = np.array([[0,0]])
                polys.append(torch.zeros([1,128,2]))
                continue
            
            ori_nodes = len(contour)
            # if ori_nodes <= 1:
            #     print(contours)
            poly = uniformsample(contour, ori_nodes * self.contour_dim)
            idx = four_idx(poly)
            poly = get_img_gt(poly, idx, t=self.contour_dim)
            polys.append(torch.tensor(poly, dtype=torch.float32)[None])

        return torch.cat(polys).to(instance_masks.device)
    
    def masks2points_with_ct(self, instance_masks: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        polys = []
        for i in range(instance_masks.size()[0]):
            mask = instance_masks[i]
            center = ct[i].cpu().numpy()
            img = mask.type(torch.uint8).cpu().numpy()
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                ares = []
                for contour in contours:
                    ares.append(cv2.contourArea(contour))
                index = torch.tensor(ares, dtype=torch.float32).argmax()
                contour = contours[index].reshape(-1,2) # n,x,y
            elif len(contours) == 1:
                contour = contours[0].reshape(-1,2) # n,x,y
            else:
                print(F"There is a non mask in sam out, sum={mask.sum()}.")
                contour = np.array([[0,0]])
                polys.append(torch.zeros([1,128,2]))
                continue
            
            ori_nodes = len(contour)
            # if ori_nodes <= 1:
            #     print(contours)
            poly = uniformsample(contour, ori_nodes * self.contour_dim)
            idx = four_idx_with_ct(poly, center)
            poly = get_img_gt(poly, idx, t=self.contour_dim)
            polys.append(torch.tensor(poly, dtype=torch.float32)[None])

        return torch.cat(polys).to(instance_masks.device)

    def train(self, mode: bool = True):
        self.refine.train(mode)

    def eval(self):
        self.refine.eval()