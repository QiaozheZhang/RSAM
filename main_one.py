import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from utils.file_utils import *
from utils.scheduler import *
from trainers import *
from args import *

def train_and_save(parser_args):
    init_file_path(parser_args)
    
    model = get_model(parser_args)
    if not parser_args.arch == "vit_exp":
        model.initialize_weights()
    
    criterion = get_criterion(parser_args)
    optimizer = get_optimizer(parser_args, model)
    parser_args.eta_min = 1e-5
    scheduler = get_scheduler(optimizer, parser_args.lr_policy)
    
    for epoch in range(parser_args.epochs):
        print("epoch: ", epoch, "lr: ", get_lr(optimizer))

        # from utils.sharpness import cumpute_gradient
        # all_grad1, all_grad2, max_abs_val, weight_num = cumpute_gradient(model, sampled_train_loader, criterion, param_names="linear.0.weight")
        # print(all_grad1, all_grad2, max_abs_val)
        
        train_acc1, train_acc5, train_acc10, reg_loss = train(
            train_loader, model, criterion, optimizer, parser_args
        )
        print(train_acc1, reg_loss)
        
        scheduler.step()

        acc1, acc5, acc10, loss = validate(
            test_loader, model, criterion, parser_args
        )
        
        # if acc1 == 100:
        #     acc_index = 1 if 'acc_index' not in locals() else acc_index + 1
    
    file_path = parser_args.file_path = (f'./results/{parser_args.dataset}/'
                                         f'{parser_args.arch}/recipe_{parser_args.recipe}_bias_{parser_args.bias}_schedu_{parser_args.lr_policy}_aug_{parser_args.aug}_fix_{parser_args.use_fix}_wp_{parser_args.warmup}_{parser_args.init}/model_compare/'
                                         f'seed_{parser_args.seed}_op_{parser_args.optimizer}_epochs_{parser_args.epochs}_lr_{parser_args.lr}_wd_{parser_args.wd}_bs_{parser_args.batch_size}_'
                                         f'reg_{parser_args.regularization}_lamb_{parser_args.lmbda}_schedu_{parser_args.lr_policy}_'
                                         f'bias_{parser_args.bias}_init_{parser_args.init}_warmup_{parser_args.warmup}')
    check_dir(f'./results/{parser_args.dataset}/{parser_args.arch}/recipe_{parser_args.recipe}_bias_{parser_args.bias}_schedu_{parser_args.lr_policy}_aug_{parser_args.aug}_fix_{parser_args.use_fix}_wp_{parser_args.warmup}_{parser_args.init}/model_compare')
    torch.save(model, file_path + '_{}.pth'.format('dense'))
    
    log_path = parser_args.file_path = (f'./results/{parser_args.dataset}/'
                                         f'{parser_args.arch}/recipe_{parser_args.recipe}_bias_{parser_args.bias}_schedu_{parser_args.lr_policy}_aug_{parser_args.aug}_fix_{parser_args.use_fix}_wp_{parser_args.warmup}_{parser_args.init}/log/'
                                         f'seed_{parser_args.seed}_op_{parser_args.optimizer}_epochs_{parser_args.epochs}_lr_{parser_args.lr}_wd_{parser_args.wd}_'
                                         f'reg_{parser_args.regularization}_lamb_{parser_args.lmbda}_schedu_{parser_args.lr_policy}_'
                                         f'bias_{parser_args.bias}_init_{parser_args.init}_warmup_{parser_args.warmup}')
    check_dir(f'./results/{parser_args.dataset}/{parser_args.arch}/recipe_{parser_args.recipe}_bias_{parser_args.bias}_schedu_{parser_args.lr_policy}_aug_{parser_args.aug}_fix_{parser_args.use_fix}_wp_{parser_args.warmup}_{parser_args.init}/log')
    print(f"log path is {log_path}")

# parser_args.dataset = 'MNIST'
# parser_args.arch = 'FC5'
# parser_args.gpu = 0
# parser_args.lr_policy = 'cosine_lr'
# parser_args.aug = False
# parser_args.init = 'kaiming_normal' # 'kaiming_normal' xavier_normal

# set_seed(parser_args.seed)

data = get_dataset(parser_args)
train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.val_loader
if parser_args.use_fix:
    train_loader = data.fix_train_loader
    # train_loader, test_loader = data.fix_train_loader, data.fix_test_loader

parser_args.sample_size = 5000
sampled_data = get_dataset(parser_args)
sampled_train_loader, sampled_val_loader, sampled_test_loader = sampled_data.train_loader, sampled_data.val_loader, sampled_data.val_loader

print("All parser arguments:")
for arg in vars(parser_args):
    print(f'{arg}: {getattr(parser_args, arg)}')
train_and_save(parser_args)
