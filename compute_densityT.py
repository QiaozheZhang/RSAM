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
from utils.density import density, check_density, compute_T
from utils.sharpness import compute_trace, compute_power_trace
from utils.utils import set_seed, save_loader, get_loader
from pathlib import Path
import json

# set_seed()

param_name_dict = {}

parser_args.sample_size = 0
data = get_dataset(parser_args)
train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.val_loader

parser_args.sample_size = 2000
sampled_data = get_dataset(parser_args)
sampled_train_loader, sampled_val_loader, sampled_test_loader = sampled_data.train_loader, sampled_data.val_loader, sampled_data.val_loader
if parser_args.use_fix:
    sampled_train_loader = sampled_data.fix_sample_train_loader  # fix_train_loader fix_sample_train_loader

criterion = get_criterion(parser_args)

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

index_list = [i for i in range(len(name_list))]
param_names_list = [name_list[i] for i in index_list]

for param_names in param_names_list:
    cp_model = copy.deepcopy(model)

    file_path = Path(parser_args.model_path).parent.parent
    sub_dir = file_path / "model_statistic"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_dir2 = file_path / f"model_statistic/DensityT/{param_names}"
    sub_dir2.mkdir(parents=True, exist_ok=True)

    file_path = Path(parser_args.model_path).resolve()  # 转为绝对路径
    file_name = file_path.stem

    compute_T(sampled_train_loader, cp_model, criterion, param_names, iter=100, n_v=1, dir=f"{sub_dir2}/{file_name}")