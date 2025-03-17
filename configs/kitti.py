from .base import commen, data, model, train, test, PROJECT_PATH
import os
import numpy as np

commen.dataset = 'kins'

data.scale = np.array([896, 384])
data.input_w, data.input_h = (896, 384)
data.scale_range = np.arange(0.4, 1.0, 0.1)

model.class_num = 7
model.heads = {'ct_hm': model.class_num, 'mask':1}
model.evolve_iters = 3

model.det_net = "YOLOv10"
model.det_weights = os.path.join(PROJECT_PATH, "network", "yolov10", "weights", "kitti.pt")

train.dataset = 'kitti_train'
train.optimizer = {'name': 'adamw', 'lr': 1e-4, 'weight_decay': 5e-4}
train.scheduler = {'name': 'MultiStepLR', 'milestones': [80, 120], 'gamma': 0.25}
train.batch_size = 16
train.num_workers = 24

test.test_rescale = 0.5
test.dataset = 'kitti_val'
test.with_nms = False

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
