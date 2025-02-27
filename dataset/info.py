
class DatasetInfo(object):
    dataset_info = {
        'coco_train': {
            'name': 'coco',
            'image_dir': 'data/coco/train2017',
            'anno_dir': 'data/coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'coco_val': {
            'name': 'coco',
            'image_dir': 'data/coco/val2017',
            'anno_dir': 'data/coco/annotations/instances_val2017.json',
            'split': 'val'
        },
        'coco_test': {
            'name': 'coco',
            'image_dir': 'data/coco/test2017',
            'anno_dir': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'sbd_train': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'sbd_val': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'kitti_train': {
            'name': 'kitti',
            'image_dir': 'data/kitti/training/image_2', 
            'anno_dir': 'data/kitti/training/instances_train.json',
            'split': 'train'
        }, 
        'kitti_val': {
            'name': 'kitti',
            'image_dir': 'data/kitti/testing/image_2', 
            'anno_dir': 'data/kitti/testing/instances_val.json', 
            'split': 'val'
        },
        'cocoa_train': {
            'name': 'cocoa',
            'image_dir': 'data/cocoa/train2014',
            'anno_dir': 'data/cocoa/COCO_amodal_train2014_with_classes_amodal.json',
            'split': 'train'
        },
        'cocoa_val': {
            'name': 'cocoa',
            'image_dir': 'data/cocoa/val2014',
            'anno_dir': 'data/cocoa/COCO_amodal_val2014_with_classes_amodal.json',
            'split': 'val'
        },
        'cityscapes_train': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'cityscapes_val': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'cityscapesCoco_val': {
            'name': 'cityscapesCoco',
            'image_dir': 'data/cityscapes/leftImg8bit/val',
            'anno_dir': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'cityscapes_test': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit', 
            'anno_dir': 'data/cityscapes/annotations/test', 
            'split': 'test'
        }
    }
