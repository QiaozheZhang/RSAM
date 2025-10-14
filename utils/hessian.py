import torch, tqdm
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def hutchinson(Hiter, model, criterion, parser_args):
    
    #params = [outputs]
    device = f'cuda:{parser_args.gpu}'
    params = []
    trace = 0.
    hessian_tr = 0
    grad_list = []


    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name:
                params.append(param)
            if 'bias' in name and param is not None:
                params.append(param)

    # print(params)
    # params = torch.tensor(params)
    grads = torch.autograd.grad(criterion, params, retain_graph=True, create_graph=True)

    for i in grads:
        grad_list.append(i)

    # calculate hessian trace
    trace_vhv = []

    if len(grad_list) > 0:
        for iii in range(Hiter):
            v = [torch.randint_like(p, high=2, device=device) for p in params]
            for v_i in v:
                v_i[v_i == 0] = -1

            Hv = torch.autograd.grad(grad_list,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=True)
            # hessian_tr = torch.trace(hess)
            hessian_tr = group_product(Hv, v).cpu().item()
            #trace_vhv.append(hessian_tr)
            trace += hessian_tr

    return trace, hessian_tr

def layer_hutchinson(Hiter, model, criterion, parser_args):
    
    #params = [outputs]
    device = f'cuda:{parser_args.gpu}'
    params = []
    trace = 0.
    hessian_tr = 0
    grad_list = []

    if 'resnet' in parser_args.arch:
        layer_name = 'conv1.0.weight'
    elif 'VGG' in parser_args.arch:
        layer_name = 'conv.0.weight'
    elif 'FC' in parser_args.arch:
        layer_name = 'linear.0.weight'

    for name, param in model.named_parameters():
        if param.requires_grad:
            if layer_name in name:
                if param.requires_grad:
                    params.append(param)

    # print(params)
    # params = torch.tensor(params)
    grads = torch.autograd.grad(criterion, params, retain_graph=True, create_graph=True)

    for i in grads:
        grad_list.append(i)

    # calculate hessian trace
    trace_vhv = []

    if len(grad_list) > 0:
        for iii in range(Hiter):
            v = [torch.randint_like(p, high=2, device=device) for p in params]
            for v_i in v:
                v_i[v_i == 0] = -1

            Hv = torch.autograd.grad(grad_list,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=True)
            # hessian_tr = torch.trace(hess)
            hessian_tr = group_product(Hv, v).cpu().item()
            #trace_vhv.append(hessian_tr)
            trace += hessian_tr

    return trace, hessian_tr

def estimate_trace(Hiter, train_loader, model, criterion, parser_args):
    total_trace, total_hessian_tr = 0, 0
    cumputed_sum = 0
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        if parser_args.gpu is not None:
            images = images.cuda(parser_args.gpu, non_blocking=True)
            target = target.cuda(parser_args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        trace, hessian_tr = hutchinson(Hiter, model, loss, parser_args)

        trace = trace/Hiter
        total_trace += trace*images.size()[0]
        total_hessian_tr += hessian_tr*images.size()[0]

        cumputed_sum += images.size()[0]

    return total_trace/cumputed_sum#, total_hessian_tr/len(train_loader)

def estimate_layer_trace(Hiter, train_loader, model, criterion, parser_args):
    total_trace, total_hessian_tr = 0, 0
    cumputed_sum = 0
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        if parser_args.gpu is not None:
            images = images.cuda(parser_args.gpu, non_blocking=True)
            target = target.cuda(parser_args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        trace, hessian_tr = layer_hutchinson(Hiter, model, loss, parser_args)

        trace = trace/Hiter
        total_trace += trace*images.size()[0]
        total_hessian_tr += hessian_tr*images.size()[0]

        cumputed_sum += images.size()[0]

    return total_trace/cumputed_sum#, total_hessian_tr/len(train_loader)