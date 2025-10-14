import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args.args_utils import parser_args
from torch.utils.data import random_split


class CIFAR10:
    def __init__(self):
        super(CIFAR10, self).__init__()

        data_root = os.path.join(parser_args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": parser_args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
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

        dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(transform_list),
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
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
            # indices = random.sample(range(len(train_dataset)), parser_args.sample_size)
            # sampler = SubsetRandomSampler(indices)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=parser_args.batch_size, sampler=sampler, **kwargs
            )
            self.fix_sample_train_loader = get_fix_train_data(self.train_loader, parser_args, 'train_sample')
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

        self.no_shuffle_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=parser_args.batch_size, shuffle=False, **kwargs
            )
        
        # self.fix_train_loader = get_fix_train_data(self.no_shuffle_loader, parser_args, 'train')
        # self.fix_test_loader = get_fix_train_data(self.val_loader, parser_args, 'test')

        # sampler = SubsetRandomSampler(range(5000))
        # self.sample_train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=parser_args.batch_size, sampler=sampler, **kwargs
        # )
        # self.fix_sample_train_loader = get_fix_train_data(self.sample_train_loader, parser_args, 'train_sample')


def get_fix_train_data(dataloader, parser_args, name):
    from torch.utils.data import TensorDataset, DataLoader
    data_pth = f'./{parser_args.data}/{parser_args.dataset}_{name}.pth'
    if not os.path.exists(data_pth):
        imgs = []
        labels = []
        for x, y in dataloader:
            imgs.append(x)
            labels.append(y)

        all_imgs = torch.cat(imgs, dim=0)
        all_labels = torch.cat(labels, dim=0)
        torch.save({'images': all_imgs, 'labels': all_labels}, data_pth)
    
    data = torch.load(data_pth)
    images = data['images']
    labels = data['labels']
    fixed_dataset = TensorDataset(images, labels)
    loader = DataLoader(fixed_dataset, batch_size=parser_args.batch_size, shuffle=True)

    return loader
