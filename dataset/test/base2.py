import torchvision
import os
import numpy as np
import cv2
from ..train.utils import augment,  transform_polys, filter_tiny_polys,get_cw_polys
import torch

from PIL import Image, ImageDraw

class Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, ann_file, data_root, split, cfg):
        super(Dataset, self).__init__(data_root, ann_file)
        self.ids = sorted(self.ids)
        self.data_root = data_root
        self.split = split
        self.cfg = cfg

    def process_info(self, img_id):
        image_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(self.data_root, image_name)
        return path, image_name

    def read_original_data(self, path):
        img = cv2.imread(path)
        return img
    
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

    def __getitem__(self, index):
        img_id = self.ids[index]
        path, image_name = self.process_info(img_id)
        img = self.read_original_data(path)
        width, height = img.shape[1], img.shape[0]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale
            )
        # orig_img = orig_img[..., ::-1].transpose((2,0,1))
        # orig_img = np.ascontiguousarray(orig_img) # contiguous
        # orig_img = torch.from_numpy(orig_img)
        # orig_img = orig_img.float()
        # print("----------------")
        # print(inp.shape)
        # print(orig_img.shape)
        # print("----------------")
        ret = {'inp': inp, 'orig_img': orig_img}

        # annids = self.coco.getAnnIds(img_id)
        # anno = self.coco.loadAnns(annids)
        # instance_polys = []
        # cls_ids = []
        # for ann in anno:
        #     tmp = ann['segmentation']
        #     polys = []
        #     for poly_tmp in tmp:
        #         polys.append(np.array(poly_tmp).reshape(-1,2))
        #     instance_polys.append(polys)
        #     cls_ids.append(ann['category_id'] - 1)
        # instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        # instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        # # test_img = Image.fromarray(np.flip(orig_img, axis=2))
        # # img_draw = ImageDraw.ImageDraw(test_img)
        # # for poly in instance_polys:
        # #     print(poly)
        # #     exit(0)
        # #     img_draw(np.array(poly).reshape(-1).tolist(), outline='blue')
        # # test_img.save("/home/wyj/work/cvwork/e2ec_main/e2ec-main/show/cityscapestest.jpg")
        # # exit(0)

        # gt_bbox = []
        # gt_cls = []
        # for i in range(len(anno)):
        #     cls_id = cls_ids[i]
        #     instance_poly = instance_polys[i]

        #     for j in range(len(instance_poly)):
        #         poly = instance_poly[j]
        #         x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        #         x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
        #         bbox = [x_min, y_min, x_max, y_max]
        #         h, w = y_max - y_min + 1, x_max - x_min + 1
        #         if h <= 1 or w <= 1:
        #             continue
        #         gt_cls.append(cls_id)
        #         gt_bbox.append(bbox)
        # ret.update({'gt_bbox':np.array(gt_bbox), 'gt_cls':np.array(gt_cls)})

        meta = {'center': center, 'img_id': img_id, 'scale': scale, 'test': '', 'img_name': image_name}
        ret.update({'meta': meta})
        return ret

    def __len__(self):
        return len(self.ids)