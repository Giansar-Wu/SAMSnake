# SAMSnake: A Generic Contour-based Instance Segmentation Network Assisted by Efficient Segment Anything Model

![kitti](kins_demo.png)

## Installation

Please see [INSTALL.md](INSTALL.md).

## Architecture
![architecture](framework.png)

## Testing

### Prepare weights
1. Download the model weights [here](The link will be published soon after we successfully upload weights to Google Drive)

2. Unzip weights to SAMSnake.
   ```
   unzip SAMSnake_weights.zip -d /path/tp/SAMSnake
   ```

### Testing on SBD

   ```
   # testing segmentation accuracy on SBD
   python test.py sbd --checkpoint /path/to/model_sbd.pth

   # testing the speed
   python test.py sbd --checkpoint /path/to/model_sbd.pth --type speed
   ```

### Testing on Cityscapes 

   ```
   # with cityscapes official evaluator
   python test.py cityscapes --checkpoint /path/to/model_cityscapes.pth

   # testing segmentation accuracy on Cityscapes with coco evaluator
   python test.py cityscapesCoco --checkpoint /path/to/model_cityscapes.pth

   # testing the speed
   python test.py cityscapesCoco \
   --checkpoint /path/to/model_cityscapes.pth --type speed

   # testing on test set, run and submit the result file
   python test.py cityscapes --checkpoint /path/to/model_cityscapes.pth \
   --dataset cityscapes_test
   ```

### Testing on COCO

   ```
   # testing segmentation accuracy on coco val set
   python test.py coco --checkpoint /path/to/model_coco.pth

   # testing the speed
   python test.py coco --checkpoint /path/to/model_coco.pth --type speed

   # testing on coco test-dev set, run and submit data/result/results.json
   python test.py coco --checkpoint /path/to/model_coco.pth --dataset coco_test
   ```

### Testing on KINS

   ```
   # testing segmentation accuracy on KINS
   python test.py kitti --checkpoint /path/to/model_kitti.pth

   # testing the speed
   python test.py kitti --checkpoint /path/to/model_kitti.pth --type speed
   ```

### Testing on COCOA

   ```
   # testing segmentation accuracy on cocoa val set
   python test.py cocoa --checkpoint /path/to/model_cocoa.pth

   # testing the speed
   python test.py cocoa --checkpoint /path/to/model_cocoa.pth --type speed
   ```

## Visualization

   ```
   # inference and visualize the images with coco pretrained model
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True

   # you can using other pretrained model, such as cityscapes 
   python visualize.py cityscapesCoco /path/to/images \
   --checkpoint /path/to/model_cityscapes.pth

   # if you want to save the visualisation, please specify --output_dir
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True \
   --output_dir /path/to/output_dir

   # visualize the results at different stage
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True --stage coarse

   # you can reset the score threshold, default is 0.3
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True --ct_score 0.1

   # if you want to filter some of the jaggedness caused by dml 
   # please using post_process
   python visualize.py coco /path/to/images \
   --checkpoint /path/to/model_coco.pth --with_nms True \
   --with_post_process True
   ```

## Training

### Training with multi GPUS

```
CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun 
--standalone \
--nnodes=${nodes_num} \
--nproc-per-node=${gpu_nums_per_node} \
train_net_ddp.py \
--config_file ${dataset} \
--bs ${batchsize} \

# the example of training sbd dataset using 2 gpus
CUDA_VISIBLE_DEVICES=0,1 torchrun \
--standalone \
--nnodes=1 \
--nproc-per-node=2 \
train_net_ddp.py \
--config_file sbd \
--bs 24
```

### Training with single GPU

```
# sbd
python train_net.py sbd --bs $batch_size

# cityscapes
python train_net.py cityscapesCoco --bs $batch_size

# coco
python train_net.py coco --bs $batch_size

# kitti
python train_net.py kitti --bs $batch_size

# cocoa
python train_net.py cocoa --bs $batch_size
```

### Training on the other dataset

If the annotations is in coco style:

1. Add dataset information to `dataset/info.py`.

2. Modify the `configs/coco.py`, reset the `train.dataset` , `model.heads['ct_hm']` and `test.dataset`. Maybe you also need to change the `train.epochs`, `train.optimizer['milestones']` and so on.

3. Train the network.

   ```
   python train_net.py coco --bs $batch_size
   ```

If  the annotations is not in coco style:

1. Prepare `dataset/train/your_dataset.py` and `dataset/test/your_dataset.py` by referring to `dataset/train/base.py` and `dataset/test/base.py`.

2. Prepare `evaluator/your_dataset/snake.py` by referring to `evaluator/coco/snake.py`.

3. Prepare `configs/your_dataset.py` and by referring to `configs/base.py`.

4. Train the network.

   ```
   python train_net.py your_dataset --bs $batch_size
   ```

## Acknowledgement

Code is largely based on [E2EC](https://github.com/zhang-tao-whu/e2ec). Thanks for their wonderful works.
