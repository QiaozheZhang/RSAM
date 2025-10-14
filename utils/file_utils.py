import os
import torch

def check_dir(dir):
    if os.path.exists(dir) != True:
        os.makedirs(dir)

def init_file_path(parser_args):
    parser_args.file_path = './results/{}/{}/model/op_{}_lr_{}_wd_{}_reg_{}_lamb_{}_schedu_{}'.format(
    parser_args.dataset, parser_args.arch, parser_args.optimizer, parser_args.lr,
        parser_args.wd, parser_args.regularization, parser_args.lmbda, parser_args.lr_policy
)

def save_model(parser_args, model, name):
    file_path = parser_args.file_path
    check_dir('./results/{}/{}/model'.format(parser_args.dataset, parser_args.arch))
    torch.save(model, file_path + '_{}.pth'.format(name))

def list_files_in_directory(directory):
    all_items = os.listdir(directory)
    
    files = [item for item in all_items if os.path.isfile(os.path.join(directory, item))]
    
    return files