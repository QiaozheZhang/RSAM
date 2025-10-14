import torch
from vit_pytorch import ViT, SimpleViT
from args.args_utils import parser_args
import torchvision.models as models

def vit_exp():
    num_classes = 10
    if parser_args.dataset == 'CIFAR100':
        num_classes = 100
    model_width = 512
    model = SimpleViT(
                image_size=32,
                patch_size=4,
                num_classes=num_classes,
                dim=model_width,
                depth=6,
                heads=16,
                mlp_dim=model_width*2
            )
    return model

def pretrain_vitb16():
    model = models.vit_b_16(pretrained=True)
    # print(model)
    num_class = 10 if parser_args.dataset=="CIFAR10" else 100
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_class)
    return model