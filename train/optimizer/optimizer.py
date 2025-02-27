import torch


# _optimizer_factory = {
#     'adam': torch.optim.Adam,
#     'sgd': torch.optim.SGD,
#     'adamw':torch.optim.AdamW
# }


def make_optimizer(net, cfg):
    optimizer_cfg = cfg.train.optimizer
    params = []
    lr = optimizer_cfg['lr']
    weight_decay = optimizer_cfg['weight_decay']

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        # if 'dla' in key:
        #     params += [{"params": [value], "lr": lr / 10, "weight_decay": weight_decay}]
        # else:
        #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if optimizer_cfg['name'] == 'adam':
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif optimizer_cfg['name'] == 'adamw':
        optimizer = torch.optim.AdamW(params, lr, weight_decay=weight_decay)
    elif optimizer_cfg['name'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    return optimizer
