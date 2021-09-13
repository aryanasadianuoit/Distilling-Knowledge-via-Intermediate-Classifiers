import torch
from torchvision import datasets,transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def get_cifar(batch_size,
              augment=True,
              cifar10_100= "cifar10",
              data_dir = "./data/",
              output_width=32,
              output_height=32,
              shuffle=True,
              num_workers=16,
              pin_memory=True):

    if cifar10_100 == "cifar10":
        # Mean  and STD for CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    elif cifar10_100 == "cifar100":
        # Mean  and STD for CIFAR100
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((output_width, output_height)),
            transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # define transforms
        valid_transform = transforms.Compose([
            transforms.Resize((output_width, output_height)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        # define transforms
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    if cifar10_100 == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform)

        valid_dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=valid_transform)

    elif cifar10_100 == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform)

        valid_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    train_sampler = SubsetRandomSampler(indices)

    if shuffle:
        np.random.shuffle(indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    if cifar10_100 == "cifar100":
            valid_loader = get_test_loader_cifar(batch_size=batch_size,
                                             dataset="cifar100",
                                             output_width=output_width,
                                             output_height=output_height)
    elif cifar10_100 == "cifar10":
            valid_loader = get_test_loader_cifar(batch_size=batch_size,
                                                 dataset="cifar10",
                                                 output_width=output_width,
                                                 output_height=output_height)
    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader
    }
    dataset_sizes = {"train": len(train_sampler),
                         "val": len(valid_dataset)}


    return data_loader_dict, dataset_sizes



def get_test_loader_cifar(
                    batch_size,
                    dataset="cifar10",
                    output_height = 32,
                    output_width = 32,
                    shuffle=True,
                    num_workers=16,
                    pin_memory=True,
                    data_dir = "./data/"):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    if dataset =="cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        # define transform
        transform = transforms.Compose([
            # transforms.Resize((output_width, output_height)),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )



    elif dataset =="cifar100":
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.267, 0.256, 0.276])

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
