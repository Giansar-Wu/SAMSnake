from network import make_network
from train.trainer.make_trainer import make_trainer
from train.optimizer.optimizer import make_optimizer
from train.scheduler.scheduler import make_lr_scheduler
from train.recorder.recorder import make_recorder
from dataset.data_loader import make_ddp_data_loader
from train.model_utils.utils import load_model, save_model, load_network
from evaluator.make_evaluator import make_evaluator
import argparse
import importlib
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
import os
import datetime
from datetime import timedelta

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file)
    if args.bs != 'None':
        cfg.train.batch_size = int(args.bs)
    # if args.dml != 'True':
    #     cfg.train.with_dml = False
    return cfg

def run(cfg):
    # dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    dist.init_process_group("nccl")
    dist.barrier()

    local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = dist.get_rank()
    # rank = dist.get_global_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    seed = local_rank + 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg.commen.init_dir()
    network = make_network.get_network(cfg)
    network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
    network = network.to(local_rank)
    trainer = make_trainer(network, cfg)
    trainer.network = DDP(trainer.network, device_ids=[local_rank], find_unused_parameters=True)
    network = trainer.network.module.net
    optimizer = make_optimizer(network, cfg)
    scheduler = make_lr_scheduler(optimizer, cfg)
    recorder = make_recorder(cfg.commen.record_dir)
    evaluator = make_evaluator(cfg)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}

    if args.type == 'finetune':
        begin_epoch = load_network(network, model_dir=args.checkpoint, map_location=map_location)
    else:
        begin_epoch = load_model(network, optimizer, scheduler, recorder, args.checkpoint, map_location=map_location)

    # train_loader, val_loader = make_ddp_data_loader(cfg=cfg, gpus=world_size)
    train_loader = make_ddp_data_loader(cfg=cfg, gpus=world_size)
    # if local_rank == 0:
    val_loader = make_ddp_data_loader(cfg=cfg, gpus=world_size, train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        train_loader.sampler.set_epoch(epoch)
        recorder.epoch = epoch
        trainer.train_ddp(trainer.network, epoch, cfg.train.epoch, train_loader, optimizer, recorder, local_rank, world_size)
        scheduler.step()
        # dist.barrier()
        # if local_rank == 0:
            # if (epoch + 1) % cfg.train.save_ep == 0:
            #     save_model(network, optimizer, scheduler, recorder, epoch, cfg.commen.model_dir)
            # if (epoch + 1) % cfg.train.eval_ep == 0:
            #     trainer.val(epoch, val_loader, evaluator, recorder)
            # save_model(network, optimizer, scheduler, recorder, epoch, cfg.commen.model_dir)
        if local_rank == 0 and  (epoch + 1) % cfg.train.eval_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.commen.model_dir)
        if (epoch + 1) % cfg.train.eval_ep == 0:
            trainer.val_ddp(trainer.network.module, epoch, val_loader, evaluator, recorder, local_rank, world_size)
        dist.barrier()

    dist.destroy_process_group()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default='coco')
    parser.add_argument("--checkpoint", default="None")
    parser.add_argument("--type", default="continue")
    parser.add_argument("--bs")
    args = parser.parse_args()
    cfg = get_cfg(args)
    run(cfg)
