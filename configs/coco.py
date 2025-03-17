from .base import commen, data, model, train, test, PROJECT_PATH
import os
import numpy as np

commen.dataset = 'coco'

# data.scale = None
data.scale = np.array([640, 640])
data.input_w, data.input_h = (640, 640)
data.test_scale = (640, 640)

model.class_num = 80
model.heads = {'ct_hm': model.class_num, 'mask':1}
model.evolve_iters = 3

model.det_net = "YOLOv10"
model.det_weights = os.path.join(PROJECT_PATH, "network", "yolov10", "weights", "yolov10m.pt")

train.batch_size = 24
train.epoch = 50
train.dataset = 'coco_train'
train.optimizer = {'name': 'adamw', 'lr': 1e-4, 'weight_decay': 5e-4}

test.dataset = 'coco_val'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test