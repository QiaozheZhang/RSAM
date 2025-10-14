import torch
import copy
import data
import models
import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    random.seed(seed)                     # Python 自带随机模块
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # CPU 上的 PyTorch 随机
    torch.cuda.manual_seed(seed)          # 单个 GPU
    torch.cuda.manual_seed_all(seed)      # 所有 GPU（多卡）
    
    torch.backends.cudnn.deterministic = True   # 确保 cudnn 可复现
    torch.backends.cudnn.benchmark = False      # 禁用自动优化，配合上面使用

def get_unseeded_generator():
    seed = int.from_bytes(os.urandom(4), byteorder="big")
    return torch.Generator().manual_seed(seed)

def save_loader(loader, path):
    all_images, all_labels = [], []
    for x, y in loader:
        all_images.append(x)
        all_labels.append(y)

    # 合并成单个 Tensor
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 保存为新数据集
    torch.save((all_images, all_labels), path)

def get_loader(path, bs):
    images, labels = torch.load(path)
    subset_dataset = TensorDataset(images, labels)
    subset_loader = DataLoader(subset_dataset, batch_size=bs, shuffle=False)

    return subset_loader

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def get_dataset(parser_args):
    print(f"=> Getting {parser_args.dataset} dataset")
    print(f"=> Dataset augmentation is {parser_args.aug}")
    print(f"=> Full training data is {parser_args.use_full_data}")
    dataset = getattr(data, parser_args.dataset)()#(parser_args)

    return dataset

def get_model(parser_args):
    print("=> Creating model '{}'".format(parser_args.arch))
    if parser_args.arch in ['FC5', 'FC12', 'FC', 'FCwide', 'FCwwide']:
        model = models.__dict__[parser_args.arch](parser_args)
    else:
        model = models.__dict__[parser_args.arch]()
    #if parser_args.fixed_init:
    #    set_seed(parser_args.seed)
    
    return model.cuda(parser_args.gpu)

def get_criterion(parser_args):
    if parser_args.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss().cuda(parser_args.gpu)
    if parser_args.criterion == 'mse':
        criterion = nn.MSELoss().cuda(parser_args.gpu)

    return criterion

def get_optimizer(args, model):
    opt_algo = args.optimizer
    opt_lr = args.lr
    opt_wd = args.wd
    
    print("=> Getting optimizer '{}'".format(opt_algo))
    if opt_algo == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if (
            "bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if (
            "bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": opt_wd,
                },
                {"params": rest_params, "weight_decay": opt_wd},
            ],
            opt_lr,
            momentum=args.momentum,
            weight_decay=opt_wd,
            nesterov=args.nesterov,
        )
    elif opt_algo == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr,
            weight_decay=opt_wd
        )
    elif opt_algo == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_lr,
            weight_decay=opt_wd
        )
    elif opt_algo == "lbfgs":
        optimizer = torch.optim.LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()), lr=opt_lr,
            max_iter=20, tolerance_grad=1e-5)

    return optimizer

def get_layers(parser_args, model):
    if parser_args.arch in ['FC', 'FC2', 'FC12', 'FC13', 'LeNet', 'VGG16', 'CNet', 'AlexNet']:
        linear_layers, conv_layers = [], []
        for layer_index in range(0, len(model.linear)):
            if isinstance(model.linear[layer_index], nn.Linear):
                linear_layers.append(model.linear[layer_index])

        for layer_index in range(0, len(model.conv)):
            if isinstance(model.conv[layer_index], nn.Conv2d):
                conv_layers.append(model.conv[layer_index])
    elif parser_args.arch in ['resnet18', 'resnet50', 'tinyResNet18', 'tinyResNet50', 'KResNet18', 'KResNet50', 'TinyResNet18', 'TinyResNet50']:
        linear_layers = [model.fc]
        conv_layers = []
        for layer in model.conv1:
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(layer)
        for layer in [model.conv2_x, model.conv3_x, model.conv4_x, model.conv5_x]:
            for basic_block_id in range(len(layer)):
                for basic_layer in layer[basic_block_id].residual_function:
                    if isinstance(basic_layer, nn.Conv2d):
                        conv_layers.append(basic_layer)
                for basic_layer in layer[basic_block_id].shortcut:
                    if isinstance(basic_layer, nn.Conv2d):
                        conv_layers.append(basic_layer)

    return linear_layers, conv_layers

def get_regularization_loss(parser_args, model, regularizer='L1', lmbda=1):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    #linear_layers, conv_layers = get_layers(parser_args, model)

    regularization_loss = torch.tensor(0.).to(f'cuda:{parser_args.gpu}')
    if regularizer == 'L2':
        for name, params in model.named_parameters():
            if ".bias" in name:
                regularization_loss += torch.norm(params, p=2)**2

            elif ".weight" in name:
                regularization_loss += torch.norm(params, p=2)**2
        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'L1':
        # reg_loss =  ||p||_1
        for name, params in model.named_parameters():
            if ".bias" in name:
                regularization_loss += torch.norm(params, p=1)

            elif ".weight" in name:
                regularization_loss += torch.norm(params, p=1)
        regularization_loss = lmbda * regularization_loss

    #print('red loss: ', regularization_loss)
    
    return regularization_loss

def prune_model(parser_args, model, sparsity_rate):
    cp_model = copy.deepcopy(model)
    linear_layers, conv_layers = get_layers(parser_args, cp_model)

    num_active_weight = 0
    num_active_biases = 0
    active_weight_list = []
    active_bias_list = []

    for layer in conv_layers+linear_layers:
        num_active_weight += torch.ones_like(layer.weight).data.sum().item()
        active_weight = layer.weight.data.clone()
        active_weight_list.append(active_weight.view(-1))

    number_of_weight_to_prune = np.ceil(
        sparsity_rate * num_active_weight).astype(int)
    number_of_biases_to_prune = np.ceil(
        sparsity_rate * num_active_biases).astype(int)

    agg_weight = torch.cat(active_weight_list)
    agg_bias = torch.tensor([])

    if number_of_weight_to_prune == 0:
        number_of_weight_to_prune = 1
    if number_of_biases_to_prune == 0:
        number_of_biases_to_prune = 1

    weight_threshold = torch.sort(
        torch.abs(agg_weight), descending=False).values[number_of_weight_to_prune-1].item()
    bias_threshold = -1

    num = 0
    for layer in conv_layers+linear_layers:
        scores = torch.gt(layer.weight.abs(), 
                           torch.ones_like(layer.weight)*weight_threshold).int()
    
        layer.weight.data = layer.weight.data * scores.data
        num += scores.data.sum().item()
    
    print('pruning done!', weight_threshold, num_active_weight*sparsity_rate, num_active_weight-num)

    return cp_model
