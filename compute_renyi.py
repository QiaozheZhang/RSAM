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
import json, math
from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10, resnet20, resnet56, resnext29_32x4d

# set_seed()

# parser_args.dataset = "MNIST"
# parser_args.arch = "FC5"

# parser_args.model_path = "/root/localfile/nas/QiaozheZhang/workspace/20250427/robustness/results/MNIST/FC5/model_compare/op_sgd_lr_0.001_wd_0.0_reg_L1_lamb_0.0_schedu_cosine_lr_bias_True_dense.pth"

if parser_args.dataset == "CIFAR10":
    num_classes=10
elif parser_args.dataset == "CIFAR100":
    num_classes=100
elif parser_args.dataset == "TinyImageNet":
    num_classes=200
if parser_args.arch in ['wrn28_2', 'wrn28_10', 'resnet20', 'resnet56', 'resnext29_32x4d']:
    model = eval(parser_args.arch)(num_classes=num_classes).cuda(parser_args.gpu)
else:
    model = get_model(parser_args)

name_list = []
for name, params in model.named_parameters():
    if 'weight' in name:
        if len(params.shape) > 1:
            name_list.append(name)
name_list.append(None)

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

    alpha = 1
    M = 15
    N = 100

    v_norm_list = torch.load(f'{sub_dir2}/{file_name}.pth', weights_only=False)['v_norm'][:N]
    T_list = torch.load(f'{sub_dir2}/{file_name}.pth', weights_only=False)['T'][:N]
    e1 = torch.zeros(M, dtype=T_list[0].dtype, device=T_list[0].device)
    e1[0] = 1.0

    T_a_list = []
    for T in T_list:
        eigvals, eigvecs = torch.linalg.eigh(T)  # T is real symmetric
        T_a = eigvecs @ torch.diag(eigvals**alpha) @ eigvecs.T  # f(T) = T^a
        # T_a = eigvecs @ torch.diag(torch.abs(eigvals)**alpha) @ eigvecs.T  # f(T) = T^a
        T_a_list.append(T_a)

    scalar_estimate_list = [v_norm**2 * torch.dot(e1, T_a @ e1) for v_norm,T_a in zip(v_norm_list,T_a_list)]
    scalar_estimate_list = [i.item() for i in scalar_estimate_list]

    Tr = np.mean(scalar_estimate_list)
    print(Tr)

    T_a_list = []
    for T in T_list:
        eigvals, eigvecs = torch.linalg.eigh(T)  # T is real symmetric
        # T_a = eigvecs @ torch.diag(eigvals**alpha) @ eigvecs.T  # f(T) = T^a
        T_a = eigvecs @ torch.diag(torch.abs(eigvals)**alpha) @ eigvecs.T  # f(T) = T^a
        T_a_list.append(T_a)

    scalar_estimate_list = [v_norm**2 * torch.dot(e1, T_a @ e1) for v_norm,T_a in zip(v_norm_list,T_a_list)]
    scalar_estimate_list = [i.item() for i in scalar_estimate_list]

    Tr_abs = np.mean(scalar_estimate_list)
    print(Tr_abs)

    list1 = [1.001, 1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
    list2 = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.0001]
    all_list = list1 + list2 #[3, 2.5, 2.1, 1.9, 1.5, 1.1, 1.01, 1.001, 0.999, 0.99, 0.9, 0.6, 0.5, 0.3, 0.1, 0.06, 0.03, 0.01, 0.0001] #list1 + list2
    for alpha in all_list:
        # if alpha < 1.1 and alpha > 0.9:
        #     Tr = Tr_abs
        # else:
        #     Tr = Tr
        # alpha = 3
        T_a_list = []
        for T in T_list:
            eigvals, eigvecs = torch.linalg.eigh(T)  # T is real symmetric
            T_a = eigvecs @ torch.diag(torch.abs(eigvals)**alpha) @ eigvecs.T  # f(T) = T^a
            T_a_list.append(T_a)

        scalar_estimate_list = [v_norm**2 * torch.dot(e1, T_a @ e1) for v_norm,T_a in zip(v_norm_list,T_a_list)]
        scalar_estimate_list = [i.item() for i in scalar_estimate_list]

        cleaned_list = scalar_estimate_list #[x for x in scalar_estimate_list if not math.isnan(x)]

        Tr_a = np.mean(cleaned_list)
        print(len(cleaned_list), Tr_a)

        try:
            renyi_entropy = 1/(1-alpha)*math.log(Tr_a/(Tr**alpha))
        except Exception as e:
            renyi_entropy = -10000000

        if 'renyi_entropy' not in result_dict:
            result_dict['renyi_entropy'] = {}
        if param_names not in result_dict['renyi_entropy']:
            result_dict['renyi_entropy'][param_names] = {}
        result_dict['renyi_entropy'][param_names][f"{alpha}"] = renyi_entropy

    # log det H
    T_a_list = []
    for T in T_list:
        eigvals, eigvecs = torch.linalg.eigh(T)  # T is real symmetric
        T_a = eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.T  # f(T) = T^a
        T_a_list.append(T_a)

    scalar_estimate_list = [v_norm**2 * torch.dot(e1, T_a @ e1) for v_norm,T_a in zip(v_norm_list,T_a_list)]
    scalar_estimate_list = [i.item() for i in scalar_estimate_list]

    cleaned_list = scalar_estimate_list #[x for x in scalar_estimate_list if not math.isnan(x)]

    Tr_a = np.mean(cleaned_list)
    log_det = np.mean(cleaned_list)
    if 'log_det' not in result_dict:
        result_dict['log_det'] = {}
    result_dict['log_det'][param_names] = log_det

    with open(f"{sub_dir}/{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    print(f'Saving json file to ==> {sub_dir}/{file_name}.json')