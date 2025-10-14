import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from utils.file_utils import *
from utils.scheduler import *
from trainers import *
from args import *
from utils.sam import SAM, ASAM, RSAM, NEWRSAM, LastRSAM, FSAM, WOWRSAM
from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10, resnet20, resnet56, resnext29_32x4d
import copy
import wandb

def save_result(parser_args, rho, eta, best_accuracy, save_dir="results"):
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名
    filename = f"{parser_args.dataset}_{parser_args.arch}.txt"
    filepath = os.path.join(save_dir, filename)

    # 保存内容
    msg = (f"mode={parser_args.sam_mode}, "
           f"strength={rho}, eta={eta}, best_top1_acc={best_accuracy}\n")

    with open(filepath, "a") as f:
        f.write(msg)

    print(f"[INFO] 已保存到 {filepath}")

def train_sam(train_loader, model, criterion, minimizer, args):
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

        if parser_args.renyi_s == 0:
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            minimizer.optimizer.step()
            minimizer.optimizer.zero_grad()
        else:
            # Ascent Step
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            minimizer.ascent_step()

            # Descent Step
            criterion(model(images), target).backward()
            minimizer.descent_step()
        
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        top10.update(acc10.item(), images.size(0))
        
    return top1.avg, top5.avg, top10.avg, loss.item()

def train_and_save(parser_args):
    init_file_path(parser_args)
    if parser_args.wandb:
        run = wandb.init(
            entity="qiaozhezhang-ailab",
            project="early stopping",
            name=f"{parser_args.sam_mode}_{parser_args.renyi_s}",
            config=parser_args,
            )
    
    if parser_args.dataset == "CIFAR10":
        num_classes=10
    elif parser_args.dataset == "CIFAR100":
        num_classes=100
    elif parser_args.dataset == "TinyImageNet":
        num_classes=200
    # num_classes=100 if parser_args.dataset == "CIFAR100" else None
    if parser_args.arch in ['wrn28_2', 'wrn28_10', 'resnet20', 'resnet56', 'resnext29_32x4d']:
        model = eval(parser_args.arch)(num_classes=num_classes).cuda(parser_args.gpu)
    else:
        model = get_model(parser_args)
    # if not parser_args.arch == "vit_exp":
    #     model.initialize_weights()
    # for n,p in model.named_parameters():
    #     print(n)
    # import time
    # time.sleep(60)
    
    criterion = get_criterion(parser_args)
    optimizer = get_optimizer(parser_args, model)
    print(parser_args.wd)
    mode = 'RSAM' if parser_args.sam_mode == "PLAIN" else parser_args.sam_mode
    rho = parser_args.renyi_s
    eta = parser_args.alpha
    
    minimizer = eval(mode)(optimizer, model, rho=rho, eta=eta)
    parser_args.eta_min = 0.0 if 'pretrain' in parser_args.arch else 1e-5
    scheduler = get_scheduler(minimizer.optimizer, parser_args.lr_policy)

    best_accuracy = 0.
    
    for epoch in range(parser_args.epochs):
        print("epoch: ", epoch, "lr: ", get_lr(optimizer))

        if "RSAM" in parser_args.sam_mode:
            parser_args.renyi_s = 0
        if parser_args.dataset == "TinyImageNet":
            if best_accuracy > 30:
                parser_args.renyi_s = rho
        else:
            if epoch > parser_args.plain_epoch: #149
                parser_args.renyi_s = rho
        
        train_acc1, train_acc5, train_acc10, reg_loss = train_sam(
            train_loader, model, criterion, minimizer, parser_args
        )
        print(train_acc1, reg_loss)
        
        scheduler.step()

        acc1, acc5, acc10, loss = validate(
            test_loader, model, criterion, parser_args
        )
        
        if best_accuracy < acc1:
           best_accuracy = acc1
           best_model = copy.deepcopy(model)
        
        print(f"renyi strength {parser_args.renyi_s} with best top acc1 {best_accuracy}")
        if parser_args.wandb:
            run.log({"test acc1": acc1, "best acc1": best_accuracy})
    
    msg = f"{parser_args.dataset}_{parser_args.arch}_{parser_args.sam_mode}_strength_{rho}_eta_{eta}_with_best_top1_acc_{best_accuracy}"
    print(msg)

    os.system(f"python3 ./notifier.py {msg}")
    save_result(parser_args, rho, eta, best_accuracy)

    file_path = parser_args.file_path = (f'./results/{parser_args.dataset}/'
                                         f'{parser_args.arch}/recipe_{parser_args.recipe}_bias_{parser_args.bias}_schedu_{parser_args.lr_policy}_aug_{parser_args.aug}_fix_{parser_args.use_fix}_wp_{parser_args.warmup}_{parser_args.init}/sam_model_compare/'
                                         f'{parser_args.sam_mode}_seed_{parser_args.seed}_op_{parser_args.optimizer}_epochs_{parser_args.epochs}_lr_{parser_args.lr}_wd_{parser_args.wd}_'
                                         f'reg_{parser_args.regularization}_lamb_{parser_args.lmbda}_schedu_{parser_args.lr_policy}_'
                                         f'bias_{parser_args.bias}_init_{parser_args.init}_warmup_{parser_args.warmup}')
    check_dir(f'./results/{parser_args.dataset}/{parser_args.arch}/recipe_{parser_args.recipe}_bias_{parser_args.bias}_schedu_{parser_args.lr_policy}_aug_{parser_args.aug}_fix_{parser_args.use_fix}_wp_{parser_args.warmup}_{parser_args.init}/sam_model_compare')
    for index in range(10):
        if os.path.exists(file_path + f'_{index}.pth'):
            pass
        else:
            torch.save(best_model, file_path + f'_{index}.pth')
            break

# set_seed(parser_args.seed)

parser_args.wandb = 1 #if parser_args.sam_mode == "NEWRSAM" else 0
data = get_dataset(parser_args)
train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.val_loader

print("All parser arguments:")
for arg in vars(parser_args):
    print(f'{arg}: {getattr(parser_args, arg)}')
train_and_save(parser_args)
# load_and_train(parser_args)