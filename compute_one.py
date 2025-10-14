from utils.file_utils import *
from utils.scheduler import *
from args import *
from trainers import *
from utils.roubust_sparse import *
from utils.hessian import estimate_trace, estimate_layer_trace
from models import *
import copy
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from utils.logging import AverageMeter
from pathlib import Path
import json

print(f"=> data augmentation is {parser_args.aug}")

parser_args.sample_size = 0
data = get_dataset(parser_args)
train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.val_loader

parser_args.sample_size = 2000
if parser_args.dataset == "TinyImageNet":
    parser_args.sample_size = 1000
sampled_data = get_dataset(parser_args)
sampled_train_loader, sampled_val_loader, sampled_test_loader = sampled_data.train_loader, sampled_data.val_loader, sampled_data.val_loader
if parser_args.use_fix:
    sampled_train_loader = sampled_data.fix_sample_train_loader  # fix_train_loader fix_sample_train_loader
criterion = get_criterion(parser_args)

haar_dict = {
    'MNIST': '/root/localfile/990pro/regulus/robustness/mnist_haar_tensor.pt',
    'CIFAR10': '/root/localfile/990pro/regulus/robustness/cifar10_haar_tensor.pt',
    'CIFAR100': '/root/localfile/990pro/regulus/robustness/cifar10_haar_tensor.pt'
}
haar_dict_bak = {
    'MNIST': './mnist_haar_tensor.pt',
    'CIFAR10': './cifar10_haar_tensor.pt',
    'CIFAR100': './cifar10_haar_tensor.pt'
}
img = {
    'MNIST': {
        'size': 1*28*28,
        'shape': [1,28,28]
        },
    'CIFAR10': {
        'size': 3*32*32,
        'shape': [3,32,32]
        },
    'CIFAR100': {
        'size': 3*32*32,
        'shape': [3,32,32]
        },
    'TinyImageNet': {
        'size': 3*64*64,
        'shape': [3,64,64]
        }
}

orho_dict = {
    'MNIST': {
        'FC': 0.8,
        'FC5': 0.8
    },
    'CIFAR10': {
        'FC': 2,
        'FC5': 0.8,
        'VGG16': 0.15,
        'resnet18': 0.2,
        'resnet34': 0.2,
        'resnet50': 0.2,
        'vit_exp': 0.2
    },
    'CIFAR100': {
        'resnet18': 0.2,
        'resnet34': 0.2,
        'resnet50': 0.2,
    },
    'TinyImageNet': {
        'resnet18': 0.2,
        'resnet34': 0.2,
        'resnet50': 0.2,
    }
}

oflag = False

if oflag:
    if os.path.exists(haar_dict[parser_args.dataset]):
        pth_path = haar_dict[parser_args.dataset]
    else:
        pth_path = haar_dict_bak[parser_args.dataset]

    mnist_haar_tensor = torch.load(pth_path)
img_size = img[parser_args.dataset]['size']
img_shape = img[parser_args.dataset]['shape']

losses = AverageMeter("Loss", ":.3f")
top1 = AverageMeter("Acc@1", ":6.2f")

oatt_losses = AverageMeter("oatt_Loss", ":.3f")
oatt_top1 = AverageMeter("oatt_Acc@1", ":6.2f")

# model_file = f'./results/{parser_args.dataset}/{parser_args.arch}/model_compare/{parser_args.model_path}'
# model_file_path = f'./results/{parser_args.dataset}/{parser_args.arch}/model_compare/'
# print(model_file_path)
# files = list_files_in_directory(model_file_path)
# for file in files:
#     file_path = model_file_path + file
#     print(file_path)

print(f'==> Loading {parser_args.model_path}')
model = torch.load(parser_args.model_path, map_location=f'cuda:{parser_args.gpu}', weights_only=False)

cp_model = copy.deepcopy(model)

for i, (images, target) in tqdm.tqdm(
    enumerate(sampled_train_loader), ascii=True, total=len(sampled_train_loader)
):
    if parser_args.gpu is not None:
        images = images.cuda(parser_args.gpu, non_blocking=True)
        target = target.cuda(parser_args.gpu, non_blocking=True)

    output = model(images)
    loss = criterion(output, target)
    acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
    # print(acc1)
    losses.update(loss.item(), images.size(0))
    top1.update(acc1.item(), images.size(0))

    if oflag:
        # oattack
        orho = orho_dict[parser_args.dataset][parser_args.arch]
        Hiter = 300
        for i in range(Hiter):
            A = mnist_haar_tensor[i].to(f'cuda:{parser_args.gpu}')
            
            att_images = images.reshape(images.size(0),img_size) @ A
            att_images = images + att_images.reshape(images.size(0),img_shape[0],img_shape[1],img_shape[2]) * orho
            output = model(att_images)
            loss = criterion(output, target)
            acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
            oatt_losses.update(loss.item(), images.size(0))
            oatt_top1.update(acc1.item(), images.size(0))

print(losses.avg, oatt_top1.avg, oatt_losses.avg, loss)

cp_model = copy.deepcopy(model)
acc1, acc5, acc10, loss = validate(
    sampled_test_loader, cp_model, criterion, parser_args
)

file_path = Path(parser_args.model_path).parent.parent
sub_dir = file_path / "model_statistic"
if "sam" in parser_args.model_path:
    sub_dir = file_path / "sam_model_statistic"
sub_dir.mkdir(parents=True, exist_ok=True)

file_path = Path(parser_args.model_path).resolve()  # 转为绝对路径
file_name = file_path.stem

nature_dict ={
            'train_loss': losses.avg,
            'train_acc': top1.avg,
            'test_loss': loss,
            'test_acc': acc1
        }

oattack_dict = {
            'train_loss': oatt_losses.avg,
            'train_acc': oatt_top1.avg,
        }

json_file = f"{sub_dir}/{file_name}.json"
if os.path.exists(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            result_dict = data
    
    result_dict['nature'] = nature_dict
    result_dict['oattack'] = oattack_dict
else:
    result_dict = {
        'nature': nature_dict,
        'oattack': oattack_dict
    }

with open(f"{sub_dir}/{file_name}.json", "w", encoding="utf-8") as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)
print(f'Saving json file to ==> {sub_dir}/{file_name}.json')