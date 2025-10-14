import os

import torch
from torchvision import datasets, transforms
from args.args_utils import parser_args

import torch.multiprocessing
from torch.utils.data.sampler import SubsetRandomSampler

torch.multiprocessing.set_sharing_strategy("file_system")


import torch
from collections import defaultdict

def make_stratified_subset_indices(dataset, N, seed=0):
    # 取标签
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [y for _, y in getattr(dataset, "samples")]
    targets = [int(t) for t in targets]

    num_classes = max(targets) + 1
    idx_per_class = defaultdict(list)
    for i, y in enumerate(targets):
        idx_per_class[y].append(i)

    g = torch.Generator().manual_seed(seed)
    base, rem = divmod(N, num_classes)

    # 决定哪些类多拿 1 张（避免用 Python 的 random）
    order = torch.randperm(num_classes, generator=g).tolist()
    take_extra = set(order[:rem])

    chosen = []
    for c in range(num_classes):
        k = base + (1 if c in take_extra else 0)
        inds = torch.tensor(idx_per_class[c])
        # 从该类中无放回采样 k 张
        perm = torch.randperm(len(inds), generator=g)[:min(k, len(inds))]
        chosen.extend(inds[perm].tolist())
    return chosen

class TinyImageNet:
    def __init__(self):
        super(TinyImageNet, self).__init__()

        sampler = None
        if parser_args.sample_size > 0:
            sampler = SubsetRandomSampler(range(parser_args.sample_size))

        #data_root = os.path.join("tiny-imagenet-200")
        data_root = os.path.join(parser_args.data, "tiny-imagenet-200")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": parser_args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")
        testdir = os.path.join(data_root, "test")

        '''
        if False: #True:
            normalize = transforms.Normalize(
                mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]
            )

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomRotation(20),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        '''
        normalize = transforms.Normalize(
                mean=[0.48024578664982126, 0.44807218089384643, 0.3975477478649648],
                std=[0.2769864069088257, 0.26906448510256, 0.282081906210584]
            )

        transform_list = [
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ] if parser_args.aug else [
                    transforms.ToTensor(),
                    normalize,
                ]
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                transform_list
            ),
        )

        if parser_args.sample_size > 0:
            # sampler = SubsetRandomSampler(range(parser_args.sample_size))
            # self.train_loader = torch.utils.data.DataLoader(
            #     train_dataset, batch_size=parser_args.batch_size, sampler=sampler, **kwargs
            # )
            indices = make_stratified_subset_indices(train_dataset, N=parser_args.sample_size, seed=1)
            # sampler = torch.utils.data.SubsetRandomSampler(indices)  # 每次迭代都会打乱顺序
            # self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parser_args.batch_size,
            #                                sampler=sampler, **kwargs)
            subset = torch.utils.data.Subset(train_dataset, indices)
            self.train_loader = torch.utils.data.DataLoader(subset, batch_size=parser_args.batch_size,
                                                    shuffle=False, **kwargs)
        else:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
            )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                testdir,  
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=parser_args.batch_size,
            shuffle=False,
            **kwargs
        )

        self.actual_val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=parser_args.batch_size,
            shuffle=False,
            **kwargs
        )