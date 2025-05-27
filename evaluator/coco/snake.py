import os
import json
from ..sbd import utils
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import torch
import numpy as np

class Evaluator:
    def __init__(self, result_dir, anno_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_dir
        self.coco = coco.COCO(ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy()

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = [utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = utils.coco_poly_to_rle(py, ori_h, ori_w)

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def miou(self, ious: dict):
        cat_ious = {}
        cat_nums = {}
        for ins_key in ious.keys():
            if len(ious[ins_key])==0:
                continue
            iou_v = np.max(np.array(ious[ins_key]))
            catid = ins_key[1]
            if catid not in cat_ious.keys():
                cat_ious[catid] = 0.0
                cat_nums[catid] = 0
            cat_ious[catid] =  (iou_v - cat_ious[catid]) / (cat_nums[catid] + 1)  + cat_ious[catid]
            cat_nums[catid] += 1
        # print(cat_ious)
        # print(cat_ious.values())
        return sum(cat_ious.values()) / len(cat_ious.values())


    def summarize(self):
        def _summarize(p, s, ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)
            return iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, s)
        
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(F"mIoU:{self.miou(coco_eval.ious)}")
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        summarize_info = []
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[0], 1))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[1], 1, iouThr=.5, maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[2], 1, iouThr=.75, maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[3], 1, areaRng='small', maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[4], 1, areaRng='medium', maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[5], 1, areaRng='large', maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[6], 0, maxDets=coco_eval.params.maxDets[0]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[7], 0, maxDets=coco_eval.params.maxDets[1]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[8], 0, maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[9], 0, areaRng='small', maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[10], 0, areaRng='medium', maxDets=coco_eval.params.maxDets[2]))
        summarize_info.append(_summarize(coco_eval.params, coco_eval.stats[11], 0, areaRng='large', maxDets=coco_eval.params.maxDets[2]))
        return coco_eval.stats[0], summarize_info

class DetectionEvaluator:
    def __init__(self, result_dir, anno_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_dir
        self.coco = coco.COCO(ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        # box = detection[:, :4].detach().cpu().numpy() * snake_config.down_ratio
        box = detection[:, :4].detach().cpu().numpy()
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        # py = output['py'][-1].detach()
        # if len(py) == 0:
        #     return 
        # box = torch.cat([torch.min(py, dim=1, keepdim=True)[0], torch.max(py, dim=1, keepdim=True)[0]], dim=1)
        # box = box.cpu().numpy()

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        if len(box) == 0:
            return

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']

        coco_dets = []
        for i in range(len(label)):
            box_ = utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            box_[2] -= box_[0]
            box_[3] -= box_[1]
            box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'bbox': box_,
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'ap': coco_eval.stats[0]}