import torch.nn as nn
from .utils import FocalLoss, DMLoss, sigmoid
import torch

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class IndL1Loss1d(nn.Module):
    def __init__(self, type='l1'):
        super(IndL1Loss1d, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction='sum')
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss

class NetworkWrapper(nn.Module):
    def __init__(self, net:torch.nn.Module, with_dml=True, with_cmask=True, with_hm=True, start_epoch=10, weight_dict=None, train_det = False):
        super(NetworkWrapper, self).__init__()
        self.with_dml = with_dml
        self.with_cmask = with_cmask
        self.with_hm = with_hm
        self.net = net
        self.ct_crit = FocalLoss()
        self.m_crit = FocalLoss()
        self.wh_crit = IndL1Loss1d('smooth_l1')
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.weight_dict = weight_dict
        self.start_epoch = start_epoch
        self.train_det = train_det

        if with_dml:
            self.dml_crit = DMLoss(type='smooth_l1')
        else:
            self.dml_crit = self.py_crit

    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        if 'test' in batch['meta']:
            return output
        epoch = batch['epoch']
        scalar_stats = {}
        loss = 0.

        keyPointsMask = batch['keypoints_mask'][batch['ct_01']]

        # cmask loss
        if self.with_cmask:
            mask_loss = self.m_crit(torch.sigmoid(output['mask']), batch['cmask'])
            scalar_stats.update({'mask_loss': mask_loss})
            loss += mask_loss

        # ct loss
        if self.with_hm:
            ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
            scalar_stats.update({'ct_loss': ct_loss})
            loss += ct_loss

        # coarse loss
        num_polys = len(output['poly_coarse'])
        if num_polys == 0:
            coarse_py_loss = torch.sum(output['poly_coarse']) * 0.
        else:
            coarse_py_loss = self.py_crit(output['poly_coarse'], output['img_gt_polys'])
        scalar_stats.update({'coarse_py_loss': coarse_py_loss})
        loss += coarse_py_loss * self.weight_dict['coarse']

        # evolution loss
        py_loss = 0
        n = len(output['py_pred']) - 1 if self.with_dml else len(output['py_pred'])
        for i in range(n):
            if num_polys == 0:
                part_py_loss = torch.sum(output['py_pred'][i]) * 0.0
            else:
                part_py_loss = self.py_crit(output['py_pred'][i], output['img_gt_polys'])
            py_loss += part_py_loss / len(output['py_pred'])
            scalar_stats.update({'py_loss_{}'.format(i): part_py_loss})
        loss += py_loss * self.weight_dict['evolve']

        if self.with_dml and epoch >= self.start_epoch and num_polys != 0:
            dm_loss = self.dml_crit(output['py_pred'][-2],
                                    output['py_pred'][-1],
                                    output['img_gt_polys'],
                                    keyPointsMask)
            scalar_stats.update({'end_set_loss': dm_loss})
            loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']
        else:
            dm_loss = torch.sum(output['py_pred'][-1]) * 0.0
            scalar_stats.update({'end_set_loss': dm_loss})
            loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']

        scalar_stats.update({'loss': loss})

        return output, loss, scalar_stats

    def train(self, mode: bool = True):
        self.net.train(mode)

    def eval(self):
        self.net.eval()