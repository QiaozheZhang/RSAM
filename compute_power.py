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
from utils.density import density, check_density
from utils.sharpness import compute_trace, compute_power_trace
from utils.utils import set_seed, save_loader, get_loader
from pathlib import Path
import json

# set_seed()

param_name_dict = {
    'MNIST': {
        'FC': "linear.0.weight",
        'FC5': "linear.0.weight"
    },
    'CIFAR10': {
        'FC': "linear.0.weight",
        'FC5': "linear.0.weight",
        'VGG16': "conv.0.weight",
        'resnet18': "conv1.0.weight"
    },
    'CIFAR100': {
        'resnet18': "conv1.0.weight"
    }
}

param_name_dict = {
    'MNIST': {
        'FC': ["linear.0.weight", "linear.2.weight", "linear.4.weight", "linear.6.weight", "linear.8.weight", None],
        'FC5': ["linear.0.weight", "linear.2.weight", "linear.4.weight", "linear.6.weight", "linear.8.weight", None]
    },
    'CIFAR10': {
        'FC': ["linear.0.weight", "linear.2.weight", "linear.4.weight", "linear.6.weight", "linear.8.weight", None],
        'FC5': ["linear.0.weight", "linear.2.weight", "linear.4.weight", "linear.6.weight", "linear.8.weight", None]
    }
}

parser_args.sample_size = 0
data = get_dataset(parser_args)
train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.val_loader

parser_args.sample_size = 2000
if parser_args.dataset == "resnet20":
    parser_args.sample_size = 1000
if parser_args.dataset == "TinyImageNet":
    parser_args.sample_size = 1000

sampled_data = get_dataset(parser_args)
sampled_train_loader, sampled_val_loader, sampled_test_loader = sampled_data.train_loader, sampled_data.val_loader, sampled_data.val_loader
if parser_args.use_fix:
    sampled_train_loader = sampled_data.fix_sample_train_loader  # fix_train_loader fix_sample_train_loader

criterion = get_criterion(parser_args)

# model_file = f'./results/{parser_args.dataset}/{parser_args.arch}/model_compare/{parser_args.model_path}'
# model_file_path = f'./results/{parser_args.dataset}/{parser_args.arch}/model_compare/'
# print(model_file_path)
# files = list_files_in_directory(model_file_path)
# for file in files:
#     file_path = model_file_path + file
#     print(file_path)

# parser_args.model_path = "/root/localfile/nas/QiaozheZhang/workspace/20250427/robustness/results/MNIST/FC5/model_compare/op_sgd_lr_0.01_wd_0.0_reg_L1_lamb_0.0_schedu_cosine_lr_bias_True_dense.pth"
print(f'==> Loading {parser_args.model_path}')
model = torch.load(parser_args.model_path, map_location=f'cuda:{parser_args.gpu}', weights_only=False)
name_list = []
for name, params in model.named_parameters():
    if 'weight' in name:
        if len(params.shape) > 1:
            name_list.append(name)
name_list.append(None)
param_name_dict[parser_args.dataset]={}
param_name_dict[parser_args.dataset][parser_args.arch] = name_list
cp_model = copy.deepcopy(model)

# if parser_args.arch == "VGG16":
#     index_list = [0]
# elif parser_args.arch == "resnet18":
#     index_list = [i for i in range(len(name_list))]
# elif parser_args.arch == "lenet":
#     index_list = [0, -2, -1]
# elif parser_args.arch == "lenetl":
#     index_list = [0, -2, -1]
# else:
if parser_args.arch == "vit_exp":
    # index_list = [i for i in range(1, len(name_list)-3)]+[len(name_list)-1]
    index_list = [23, 24]
else:
    index_list = [0, 1, 6, 13, -2, -1]
    # index_list = [i for i in range(len(name_list))]
param_names_list = [name_list[i] for i in index_list]
if "sam" in parser_args.model_path:
    param_names_list = [None]


for param_names in param_names_list:
    cp_model = copy.deepcopy(model)
    # param_names = param_name_dict[parser_args.dataset][parser_args.arch]
    # param_names = parser_args.param_names
    # param_names = None

    file_path = Path(parser_args.model_path).parent.parent
    sub_dir = file_path / "model_statistic"
    if "sam" in parser_args.model_path:
        sub_dir = file_path / "sam_model_statistic"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_dir2 = file_path / f"model_statistic/Lanczos/{param_names}"
    if "sam" in parser_args.model_path:
        sub_dir2 = file_path / f"sam_model_statistic/Lanczos/{param_names}"
    sub_dir2.mkdir(parents=True, exist_ok=True)

    file_path = Path(parser_args.model_path).resolve()  # 转为绝对路径
    file_name = file_path.stem
    json_file = f"{sub_dir}/{file_name}.json"

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            result_dict = data

    alpha = 0.1
    M = 15
    N = 100
    if parser_args.dataset == "TinyImageNet":
        N = 100
    check_density(sampled_train_loader, cp_model, criterion, param_names, iter=M, n_v=N, a=alpha, min_iter=True, dir=f"{sub_dir2}/{file_name}")
    # ftrace = compute_power_trace(sampled_train_loader, cp_model, criterion, param_names, n_iters=500, tol=1e-4, a=alpha, window_size=20)
    # print(ftrace)

    v_norm_list = torch.load(f'{sub_dir2}/{file_name}.pth', weights_only=False)['v_norm'][:N]
    T_list = torch.load(f'{sub_dir2}/{file_name}.pth', weights_only=False)['T'][:N]
    e1 = torch.zeros(M, dtype=T_list[0].dtype, device=T_list[0].device)
    e1[0] = 1.0

    T_a_list = []
    for T in T_list:
        eigvals, eigvecs = torch.linalg.eigh(T)  # T is real symmetric
        T_a = eigvecs @ torch.diag(eigvals**alpha) @ eigvecs.T  # f(T) = T^a
        T_a_list.append(T_a)

    scalar_estimate_list = [v_norm**2 * torch.dot(e1, T_a @ e1) for v_norm,T_a in zip(v_norm_list,T_a_list)]
    scalar_estimate_list = [i.item() for i in scalar_estimate_list]

    Tr_a = np.mean(scalar_estimate_list)
    print(Tr_a)

    # ftrace = compute_power_trace(sampled_train_loader, cp_model, criterion, param_names, n_iters=200, tol=1e-4, a=alpha, window_size=20)
    # print(ftrace)

    if 'Tr(alpha)' not in result_dict:
        result_dict['Tr(alpha)'] = {}
    if param_names not in result_dict['Tr(alpha)']:
        result_dict['Tr(alpha)'][param_names] = {}
    result_dict['Tr(alpha)'][param_names][f"{alpha}"] = Tr_a

    # if f"{M}" not in result_dict['miu_list']:
    #     result_dict['miu_list'][f"{M}"] = {}
    # result_dict['miu_list'][f"{M}"][f"{N}"] = miu_list

    with open(f"{sub_dir}/{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    print(f'Saving json file to ==> {sub_dir}/{file_name}.json')