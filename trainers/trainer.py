import time
import torch
import tqdm
import copy
import pdb

from utils.eval_utils import accuracy
from utils.utils import get_regularization_loss
from utils.logging import AverageMeter
from utils.sharpness import get_params_grad
from regularizer.renyi_reg import renyi_reg, renyi_loss, fisher_approximation

from torch import optim
import torch.nn.functional as F

def train(train_loader, model, criterion, optimizer, args):
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    top10 = AverageMeter("Acc@10", ":6.2f")
    
    # switch to train mode
    model.train()

    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if args.arch == 'alexnet':
            images = F.interpolate(images, scale_factor=7)

        output = model(images)
        
        loss = criterion(output, target)
        
        regularization_loss = torch.tensor(0).to(f'cuda:{args.gpu}')
        
        #if args.regularization:
        regularization_loss =\
            get_regularization_loss(args, model, regularizer=args.regularization,
                                        lmbda=args.lmbda)
        
        #print('regularization_loss: ', regularization_loss)
        loss += regularization_loss
        
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        top10.update(acc10.item(), images.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return top1.avg, top5.avg, top10.avg, loss.item()

def train_with_renyi(train_loader, model, criterion, optimizer, args):
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    top10 = AverageMeter("Acc@10", ":6.2f")
    
    # switch to train mode
    model.train()

    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if args.arch == 'alexnet':
            images = F.interpolate(images, scale_factor=7)

        output = model(images)
        loss = criterion(output, target)
        
        # regularization_loss = torch.tensor(0).to(f'cuda:{args.gpu}')
        
        #if args.regularization:
        # regularization_loss =\
        #     get_regularization_loss(args, model, regularizer=args.regularization,
        #                                 lmbda=args.lmbda)
        
        if args.renyi:
            # layer_name_list = ["conv5_x.0.residual_function.0.weight", "conv5_x.0.residual_function.3.weight", "conv5_x.0.shortcut.0.weight", "conv5_x.1.residual_function.0.weight", "conv5_x.1.residual_function.3.weight", "fc.weight"]
            layer_name_list = [None]
            # renyi = renyi_loss(images, target, model, criterion, layer_name_list, n_iters=5, alpha=2)
            renyi = fisher_approximation(loss, model, layer_name_list, alpha=2)
            loss += args.renyi_s*renyi
            # print(renyi)
            # regularization_loss += 0.1*renyi
            # # check gradient
            # import copy
            # cp_model = copy.deepcopy(model)
            # cp_model.zero_grad()
            # renyi2 = renyi_reg(images, target, cp_model, criterion, layer_name_list, n_iters=5, alpha=2)
            # renyi2.backward()
            # success = True
            # for name, param in cp_model.named_parameters():
            #     if param.grad is None:
            #         print(f"[FAIL] 参数 {name} 的 grad 是 None ❌")
            #         success = False
            #     elif torch.all(param.grad == 0):
            #         print(f"[WARN] 参数 {name} 的 grad 全为 0 ⚠️（可能是某些特征未被激活）")
            #     else:
            #         print(f"[OK] 参数 {name} 的 grad 正常 ✅")

            # if success:
            #     print("\n🎉 梯度计算成功，可以继续优化！")
            # else:
            #     print("\n❌ 部分梯度未被计算，检查模型结构或 loss 是否有问题")
        
        #print('regularization_loss: ', regularization_loss)
        
        
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        top10.update(acc10.item(), images.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return top1.avg, top5.avg, top10.avg, loss.item()


def validate(val_loader, model, criterion, args):
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    top10 = AverageMeter("Acc@10", ":6.2f", write_val=False)
    
    model.eval()

    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if args.arch == 'alexnet':
                images = F.interpolate(images, scale_factor=7)

            output = model(images)

            loss = criterion(output, target)

            acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            top10.update(acc10.item(), images.size(0))

    print("Model top1 Accuracy: {}".format(top1.avg))
    return top1.avg, top5.avg, top10.avg, losses.avg

def LBFGS_train(train_loader, model, criterion, args):
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=100, tolerance_grad=1e-7)
    
    # switch to train mode
    model.train()

    def closure():
        optimizer.zero_grad()
        total_loss = torch.tensor(0).to(f'cuda:{args.gpu}')

        for i, (images, target) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
        ):

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if args.arch == 'alexnet':
                images = F.interpolate(images, scale_factor=7)

            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            # total_loss += loss.item() * images.size(0)
            return loss # torch.tensor(total_loss / len(train_loader.dataset), requires_grad=True)
    
        optimizer.step(closure)