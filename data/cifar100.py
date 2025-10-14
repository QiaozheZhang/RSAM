import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args.args_utils import parser_args
from torch.utils.data import random_split


class CIFAR100:
    def __init__(self):
        super(CIFAR100, self).__init__()

        data_root = os.path.join(parser_args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": parser_args.workers, "pin_memory": True} if use_cuda else {}

        num_classes = 100

        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )
        transform_list = [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ] if parser_args.aug else [
                    transforms.ToTensor(),
                    normalize,
                ]

        transform_train = transforms.Compose(transform_list)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transform_train,
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transform_test,
        )

        if parser_args.use_full_data:
            train_dataset = dataset
            # use_full_data => we are not tuning hyperparameters
            validation_dataset = test_dataset
        else:
            val_size = 5000
            train_size = len(dataset) - val_size
            train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

        if parser_args.sample_size > 0:
            sampler = SubsetRandomSampler(range(parser_args.sample_size))
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=parser_args.batch_size, sampler=sampler, **kwargs
            )
        else:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
            )

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=parser_args.batch_size, shuffle=False, **kwargs
        )

        self.actual_val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
        )
