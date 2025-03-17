from .base import commen, data, model, train, test, PROJECT_PATH
import os
commen.dataset = 'sbd'

data.scale = None
data.test_scale = (512, 512)

model.class_num = 20
model.heads = {'ct_hm': model.class_num, 'mask':1}
model.evolve_iters = 3
model.use_normalization = True
train.with_dml = True

train.with_cmask = True
train.with_hm = True 

model.det_net = "YOLOv10"
model.det_weights = os.path.join(PROJECT_PATH, "network", "yolov10", "weights", "sbd.pt")

train.batch_size = 20
train.dataset = 'sbd_train'
train.optimizer = {'name': 'adamw', 'lr': 1e-4, 'weight_decay': 5e-4}

test.dataset = 'sbd_val'
test.with_nms = False

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test   