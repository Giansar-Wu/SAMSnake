import torch
import os
import torch.nn.functional
from termcolor import colored

def load_model(net, optim, scheduler, recorder, model_path, map_location=None):
    strict = True

    if not os.path.exists(model_path):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_path))
    if map_location is None:
        pretrained_model = torch.load(model_path, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                               'cuda:2': 'cpu', 'cuda:3': 'cpu',
                                                               'cuda:4': 'cpu', 'cuda:5': 'cpu',
                                                               'cuda:6': 'cpu', 'cuda:7': 'cpu'
                                                               })
    else:
        pretrained_model = torch.load(model_path, map_location=map_location)

    net_weight = net.state_dict()
    for key in net_weight.keys():
        if ('detector' in key) or ('efficientsam' in key):
            continue
        else:
            net_weight.update({key: pretrained_model['net'][key]})
            
        # if key not in pretrained_model['net'].keys():
        #     key_ = key.replace('conv_offset', 'conv_offset_mask')
        # else:
        #     key_ = key
        # net_weight.update({key: pretrained_model['net'][key_]})
    
    net.load_state_dict(net_weight, strict=strict)
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1

def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    net_parameters = {}
    for key, v in net.state_dict().items():
        if ("detector" not in key) and ("efficientsam" not in key):
            net_parameters.update({key:v})
    torch.save({
        'net': net_parameters,
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))
    # torch.save({
    #     'net': net.state_dict(),
    #     'optim': optim.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'recorder': recorder.state_dict(),
    #     'epoch': epoch
    # }, os.path.join(model_dir, '{}.pth'.format(epoch)))
    return

def save_weight(net, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
    }, os.path.join(model_dir, '{}.pth'.format('final')))
    return

def load_network(net, model_dir, strict=True, map_location=None):

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_dir))
    if map_location is None:
        pretrained_model = torch.load(model_dir, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                               'cuda:2': 'cpu', 'cuda:3': 'cpu',
                                                               'cuda:4': 'cpu', 'cuda:5': 'cpu',
                                                               'cuda:6': 'cpu', 'cuda:7': 'cpu'
                                                               })
    else:
        pretrained_model = torch.load(model_dir, map_location=map_location)
    if 'epoch' in pretrained_model.keys():
        epoch = pretrained_model['epoch'] + 1
    else:
        epoch = 0
    pretrained_model = pretrained_model['net']

    net_weight = net.state_dict()
    for key in net_weight.keys():
        if ('detector' in key) or ('efficientsam' in key):
            continue
        else:
            net_weight.update({key: pretrained_model[key]})
        # if key not in pretrained_model.keys():
        #     key_ = key.replace('conv_offset', 'conv_offset_mask')
        # else:
        #     key_ = key
        # net_weight.update({key: pretrained_model[key_]})

    net.load_state_dict(net_weight, strict=strict)
    return epoch
