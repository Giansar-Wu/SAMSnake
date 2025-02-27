from torch.optim import lr_scheduler
from collections import Counter


# def make_lr_scheduler(optimizer, config):
#     scheduler = MultiStepLR(optimizer, milestones=config.train.optimizer['milestones'],
#                             gamma=config.train.optimizer['gamma'])
#     return scheduler

def make_lr_scheduler(optimizer, config):
    if config.train.scheduler['name'] == 'MultiStepLR':
        print("using multisetplr!")
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.train.scheduler['milestones'], gamma=config.train.scheduler['gamma'])
    elif config.train.scheduler['name'] == 'CosineAnnealingLR':
        print("using CosineAnnealingLR!")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.scheduler['T_max'], eta_min=config.train.scheduler['eta_min'])
    return scheduler


def set_lr_scheduler(scheduler, config):
    scheduler.milestones = Counter(config.train.optimizer['milestones'])
    scheduler.gamma = config.train.optimizer['gamma']

