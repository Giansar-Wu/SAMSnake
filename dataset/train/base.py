import math
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from .douglas import Douglas
from .utils import transform_polys, filter_tiny_polys, get_cw_polys, gaussian_radius, draw_umich_gaussian,\
uniformsample, four_idx, get_img_gt, img_poly_to_can_poly, augment, polygon_to_cmask

class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, cfg):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.split = split

        self.coco = COCO(anno_file)
        self.anns = np.array(sorted(self.coco.getImgIds()))
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_continuous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.d = Douglas()

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = filter_tiny_polys(instance)
            polys = get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    # wh, ct_cls, ct_ind = []
    # ct_hm = output_size (input 4 dowm sample)
    # box, poly  = output_size (input 4 dowm sample)
    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

        # img_gt_polys = [] 绝对位置
        # keyPointsMask = []
        # can_gt_polys = [] 归一化的偏移量
    def prepare_evolution(self, poly, img_gt_polys, can_gt_polys, keyPointsMask):
        # uniformsample后并不区分上下左右点，点都是随机排列的
        img_gt_poly = uniformsample(poly, len(poly) * self.cfg.data.points_per_poly)
        
        # 下 右 上 左 4个值的索引index
        idx = four_idx(img_gt_poly)

        # 得到顺序排列的poly
        # index: begin (0),,,,,,right (32),,,,,,top (64),,,,,,,left (96)
        img_gt_poly = get_img_gt(img_gt_poly, idx)

        # poly减去最小值，进行归一化，可改进
        can_gt_poly = img_poly_to_can_poly(img_gt_poly)
        
        # Douglas算法后得到的关键轮廓点，用于DML
        key_mask = self.get_keypoints_mask(img_gt_poly)
        keyPointsMask.append(key_mask)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def get_keypoints_mask(self, img_gt_poly):
        key_mask = self.d.sample(img_gt_poly)
        return key_mask

    def __getitem__(self, index):
        data_input = {}

        ann = self.anns[index]
        anno, image_path, image_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, image_path)
        width, height = img.shape[1], img.shape[0]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)

        #detection
        output_h, output_w = inp_out_hw[2:]
        # output_size
        ct_hm = np.zeros([len(self.json_category_id_to_continuous_id), output_h, output_w], dtype=np.float32)
        ct_cls = []
        wh = []
        ct_ind = []
        gt_bbox = []
        #segmentation
        img_gt_polys = []
        can_gt_polys = []
        keyPointsMask = []

        cmask = polygon_to_cmask(instance_polys, output_h, output_w)[np.newaxis,:,:]

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                gt_bbox.append(bbox)
                self.prepare_evolution(poly, img_gt_polys, can_gt_polys, keyPointsMask)
        
        # 变换到input_size后的处理后的image (0~1)
        data_input.update({'inp': inp})

        data_input.update({'cmask': cmask})

        # 未变换的原始图像,size为input_size
        data_input.update({'orig_img': orig_img})
        # 4倍下采样后形状的输出
        # ct_cls类别id, ct_ind中心点编码（与output_h有关）
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'bbox':gt_bbox}
        
        # img_gt_polys： 得到顺序排列的poly
        # index: begin (0),,,,,,right (32),,,,,,top (64),,,,,,,bottom (96)
        
        # can_gt_polys: 归一化的poly
        # poly减去最小值，进行归一化，可改进

        # keypoints_mask: Douglas算法后得到的关键轮廓点，用于DML
        evolution = {'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys}
        data_input.update(detection)
        data_input.update(evolution)
        data_input.update({'keypoints_mask': keyPointsMask})
        ct_num = len(ct_ind) #中心点数量
        # center， scale， 图像变换参数
        # ann,图片对应的所有annotation
        meta = {'center': center, 'scale': scale, 'img_id': image_id, 'ann': ann, 'ct_num': ct_num}
        data_input.update({'meta': meta})
        return data_input

    def __len__(self):
        return len(self.anns)