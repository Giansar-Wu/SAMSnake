from .base import commen, data, model, train, test, PROJECT_PATH
import os
import numpy as np

commen.dataset = 'cityscapes'

data.scale = np.array([800, 800])
data.input_w, data.input_h = (800, 800)

model.class_num = 8
model.heads = {'ct_hm': model.class_num, 'mask':1}
model.evolve_iters = 3

model.det_net = "YOLOv10"
model.det_weights = os.path.join(PROJECT_PATH, "network", "yolov10", "weights", "cityscapes.pt")


train.dataset = 'cityscapes_train'
train.batch_size = 16
train.num_workers = 16
train.epoch = 200
train.optimizer = {'name': 'adamw', 'lr': 1e-4, 'weight_decay': 5e-4}

test.dataset = 'cityscapes_val'
test.with_nms = False

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
