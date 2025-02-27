import torch.nn as nn
from .backbone.dla import DLASeg
from .backbone.DLA34 import DLA_Encoder
from .detector_decode.refine_decode import Decode
from .evolve.evolve import Evolution
import torch
from collections import OrderedDict

class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.commen.down_ratio
        heads = cfg.model.heads
        self.test_stage = cfg.test.test_stage

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn)
        
        ### load centernet pre_trained pth
        pretrained_model = torch.load('weights/kins.pth')
        pretrained_dict = pretrained_model['net']
        dla_dict = OrderedDict((key.replace('dla.', ''), value) for key,value in pretrained_dict.items() if key.startswith('dla'))
        self.dla.load_state_dict(dla_dict,strict = True)     
        print("load centernet pre_trained pth success")
        for param in self.dla.parameters():
            param.requires_grad = False
        print("set centernet requires_grad to False")

        # for i, param in enumerate(self.dla.parameters()):
        #     param.requires_grad = True
        # for i, (name, param) in enumerate(self.dla.named_parameters()):
        #     if i<=2:
        #         print(name)
        #         print(param.requires_grad)
        # for i, param in enumerate(self.dla.parameters()):
        #     param.requires_grad = False
        # for i, (name, param) in enumerate(self.dla.named_parameters()):
        #     if i<=2:
        #         print(name)
        #         print(param.requires_grad)
        # exit(0)

        ### load centernet pre_trained pth

        self.dla_encoder = DLA_Encoder('dla{}'.format(num_layers),
                          pretrained=True,
                          down_ratio=down_ratio,
                          last_level=5,
                          )

        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score)
        
        self.gcn = Evolution(evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio)

    def forward(self, x, batch=None):
        if 'test' not in batch['meta']:
            cnn_feature = self.dla_encoder(x)
            output = {}
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                output, _ = self.dla(x)
                cnn_feature = self.dla_encoder(x)
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=ignore)
        output = self.gcn(output, cnn_feature, batch, test_stage=self.test_stage)
        return output

def get_network(cfg):
    network = Network(cfg)
    return network
