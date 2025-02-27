import torch.nn as nn
from .backbone.dla import DLASeg
from .detector_decode.refine_decode import Decode
from .evolve.evolve import Evolution
import torch
from .detector_decode.utils import decode_ct_hm, clip_to_image

from PIL import Image

class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()
        down_ratio = cfg.commen.down_ratio
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        heads = cfg.model.heads
        self.det_net = cfg.model.det_net
        self.test_stage = cfg.test.test_stage
        self.min_ct_score = cfg.test.ct_score

        self.dla = DLASeg('dla{}'.format(num_layers), 
                          heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, 
                          use_dcn=cfg.model.use_dcn)

        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, 
                                    init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, 
                                    down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score
                                    )
        
        self.gcn = Evolution(evole_ietr_num=cfg.model.evolve_iters, 
                             evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio,
                             num_points=cfg.commen.points_per_poly,
                             use_normal=cfg.model.use_normalization)
        
        if self.det_net == "YOLOv10":
            from .yolov10.ultralytics import YOLOv10
            # self.detector = YOLOv10(cfg.model.det_cfg).load(cfg.model.det_weights)
            self.detector = YOLOv10(cfg.model.det_weights)
            for param in self.detector.model.parameters():
                param.requires_grad = False
        elif self.det_net == "YOLOv8":
            from .yolov8.ultralytics import YOLO
            # self.detector = YOLO(cfg.model.det_cfg).load(cfg.model.det_weights)
            self.detector = YOLO(cfg.model.det_weights)
        else:
            print(F"There is no '{self.det_net}' detector")
            exit(0)

    def forward(self, x:torch.Tensor, batch_info_dict: dict=None):
        """_summary_

        Args:
            x (torch.Tensor): (N, C, H, W)
            batch_info_dict (dict, optional): dict_keys(['inp', 'orig_img', 'meta', 'bbox', 'wh', 'ct_hm', 'ct_cls', 'ct_ind', 'ct_01', 'ct_img_idx', 'img_gt_polys', 'can_gt_polys', 'keypoints_mask', 'epoch'])
                name:inp, size:torch.Size([4, 3, 512, 512])
                name:orig_img, size:torch.Size([4, 512, 512, 3])
                name:meta,dict_keys(['center', 'scale', 'img_id', 'ann', 'ct_num'])
                name:bbox, size:torch.Size([4, 7, 4])
                name:wh, size:torch.Size([4, 7, 2])
                name:ct_hm, size:torch.Size([4, 20, 128, 128])
                name:ct_cls, size:torch.Size([4, 7])
                name:ct_ind, size:torch.Size([4, 7])
                name:ct_01, size:torch.Size([4, 7])
                name:ct_img_idx, size:torch.Size([4, 7])
                name:img_gt_polys, size:torch.Size([4, 7, 128, 2])
                name:can_gt_polys, size:torch.Size([4, 7, 128, 2])
                name:keypoints_mask, size:torch.Size([4, 7, 128])

        Returns:
            _type_: _description_
        """
        if 'test' not in batch_info_dict['meta']:
            output, cnn_feature = self.dla(x)
            self.train_decoder(batch_info_dict, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                output, cnn_feature = self.dla(x)
            self.detection_decode(output, x, batch_info_dict, min_ct_score=self.min_ct_score)
            if self.test_stage == 'init':
                ignore = True
            else:
                ignore = False
            self.train_decoder(batch_info_dict, cnn_feature, output, is_training=False, ignore_gloabal_deform=ignore)
        output = self.gcn(output, cnn_feature, batch_info_dict, test_stage=self.test_stage)
        return output
    
    def detection_decode(self, output: dict, x: torch.Tensor, batch_info_dict: dict=None, K: int=100, min_ct_score: float=0.05):
        if self.det_net == "CenterNet":
            tmp, cnn_feature = self.detector(x)
            hm_pred, wh_pred = tmp['ct_hm'], tmp['wh']
            ct, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred, K=K) # ct [B, 100,2]  detection [B, 100,6]
            valid = detection[0, :, 4] >= min_ct_score
            ct, detection = ct[0][valid], detection[0][valid] # [filter_num, ...]
            detection[..., :4] = clip_to_image(detection[..., :4], cnn_feature.size(2), cnn_feature.size(3))
            detection[..., :4] *= 4 # upsample to input_size
            ct *= 4
        elif self.det_net == "YOLOv8":
            img = batch_info_dict['orig_img'][0].flip(dims=[2]).permute(2,0,1).div(255)[None]
            img_size = [img.size()[2], img.size()[3]]
            if max(img_size) < 640:
                img_size = 640
            result = self.detector.predict(img, verbose=False, conf=min_ct_score, max_det=K, imgsz=img_size)
            detection = result[0].boxes.data
            cx = (detection[..., 2] + detection[..., 0]) / 2
            cy = (detection[..., 3] + detection[..., 1]) / 2
            ct = torch.cat([cx[..., None], cy[..., None]], dim=1).type(torch.float32)           
        elif self.det_net == "YOLOv10":
            img = batch_info_dict['orig_img'][0].flip(dims=[2]).permute(2,0,1).div(255)[None]
            img_size = (img.size()[2], img.size()[3])
            # print(img_size)
            # print(x.size())
            # # exit(0)
            if max(img_size) < 640:
                img_size = 640
            result = self.detector.predict(img, verbose=False, conf=min_ct_score, max_det=K, imgsz=img_size)
            detection = result[0].boxes.data
            cx = (detection[..., 2] + detection[..., 0]) / 2
            cy = (detection[..., 3] + detection[..., 1]) / 2
            ct = torch.cat([cx[..., None], cy[..., None]], dim=1).type(torch.float32)
        else:
            print(F"There is no '{self.det_net}' detector")
            exit(0)

        output.update({'ct': ct, 'detection': detection})
    
    def train(self, mode: bool = True):
        self.dla.train(mode)
        self.train_decoder.train(mode)
        self.gcn.train(mode)

    def eval(self):
        self.dla.eval()
        self.train_decoder.eval()
        self.gcn.eval()

def get_network(cfg):
    network = Network(cfg)
    return network
