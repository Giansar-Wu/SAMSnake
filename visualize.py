from network import make_network
import tqdm
import torch
import os
import nms
import post_process
from dataset.data_loader import make_demo_loader
from train.model_utils.utils import load_network
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

from PIL import Image
from torchvision.transforms import Resize 

parser = argparse.ArgumentParser()

parser.add_argument("config_file", help='/path/to/config_file.py')
parser.add_argument("image_dir", help='/path/to/images')
parser.add_argument("--checkpoint", default='', help='/path/to/model_weight.pth')
parser.add_argument("--ct_score", default=0.2, help='threshold to filter instances', type=float)
parser.add_argument("--with_nms", default=False, type=bool,
                    help='if True, will use nms post-process operation', choices=[True, False])
parser.add_argument("--with_post_process", default=False, type=bool,
                    help='if True, Will filter out some jaggies', choices=[True, False])
parser.add_argument("--stage", default='final-dml', help='which stage of the contour will be generated',
                    choices=['init', 'coarse', 'final', 'final-dml'])
parser.add_argument("--output_dir", default='None', help='/path/to/output_dir')
parser.add_argument("--device", default=0, type=int, help='device idx')

args = parser.parse_args()

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file).config
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.test_stage = args.stage
    cfg.test.ct_score = args.ct_score
    return cfg

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img

class Visualizer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def visualize_ex(self, output, batch, save_dir=None):
        inp = bgr_to_rgb(unnormalize_img(batch['inp'][0], self.cfg.data.mean,
                                         self.cfg.data.std).permute(1, 2, 0))
        # print(output.keys())
        # ex = output['py']
        # print(output['poly_coarse'].size())
        # print(len(output['py']))
        # print(output['py'][0].size())
        # ex = ex[-1] if isinstance(ex, list) else ex
        # ex = output['init_poly']
        # ex = output['poly_coarse']
        ex = output['py'][-1]
        ex = ex.detach().cpu().numpy()
        # print(inp.shape)
        inp2 = torch.zeros_like(inp, dtype=torch.uint8)
        # exit(0)

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)
        # fig2, ax2 = plt.subplots(1, figsize=(20, 10))
        # ax2.axis('off')
        # ax2.imshow(inp2)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        # np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, lw=4)
            # ax2.plot(poly[:, 0], poly[:, 1], color=color, lw=4)
        if save_dir is not None:
            fig.savefig(fname=save_dir, dpi=300, bbox_inches='tight', pad_inches=0)
            # fig2.savefig(fname=F"{save_dir.split('.')[0]}_contour.jpg", dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            print("please input save_dir!")
            return 0

    def visualize(self, output, batch):
        if args.output_dir != 'None':
            file_name = os.path.join(args.output_dir, batch['meta']['img_name'][0])
        else:
            file_name = None
        self.visualize_ex(output, batch, save_dir=file_name)

def run_visualize(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint)
    network.eval()

    data_loader = make_demo_loader(args.image_dir, cfg=cfg)
    visualizer = Visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        if args.with_post_process:
            post_process.post_process(output)
        if args.with_nms:
            nms.post_process(output)
        # ct_hm = torch.sigmoid(output['ct_hm'])
        # dog = ct_hm[0,16].unsqueeze(0).unsqueeze(0)
        # img_o = batch['orig_img'][0].flip(dims=[2])
        # torch_resize = Resize(img_o.size()[0:2])
        # print(F"heatmap size :{dog.size()}")
        # print(F"img size:{img_o.size()}")
        # ht_map = torch_resize(dog)
        # print(F"heatmap size2 :{ht_map.size()}")
        # plt.imshow(img_o.cpu().numpy())
        # plt.imshow(ht_map[0,0].cpu().numpy(), alpha=0.5)
        # plt.axis('off')
        # plt.savefig("./show/heatmap2.png", dpi=300, bbox_inches='tight', pad_inches = 0.0)
        
        visualizer.visualize(output, batch)

if __name__ == "__main__":
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    run_visualize(cfg)
