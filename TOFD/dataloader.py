import torch
from torchvision import datasets,transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os

random_pad_size = 2

def get_train_valid_loader_cifars(batch_size,
                                  augment=True,
                                  cifar10_100= "cifar10",
                                  data_dir = "./data/",
                                  output_width=32,
                                  output_height=32,
                                  valid_size=0.1,
                                  shuffle=True,
                                  show_sample=False,
                                  num_workers=16,
                                  pin_memory=True,
                                  test_as_valid =True):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg


    if cifar10_100 == "cifar10":
        # Mean  and STD for CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    elif cifar10_100 == "cifar100":
        # Mean  and STD for CIFAR100
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((output_width, output_height)),
            transforms.RandomCrop(32, padding=4,padding_mode= "reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # define transforms
        valid_transform = transforms.Compose([
            transforms.Resize((output_width, output_height)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        # define transforms
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    if cifar10_100 == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        if test_as_valid:

            valid_dataset = datasets.CIFAR100(
                root=data_dir, train=False,
                download=True, transform=valid_transform,
            )

        else:
            valid_dataset = datasets.CIFAR100(
                root=data_dir, train=True,
                download=True, transform=valid_transform,
            )
    elif cifar10_100 == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        if test_as_valid:

            valid_dataset = datasets.CIFAR10(
                root=data_dir, train=False,
                download=True, transform=valid_transform,
            )

        else:
            valid_dataset = datasets.CIFAR10(
                root=data_dir, train=True,
                download=True, transform=valid_transform,
            )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    # print("spit ===> ",split,"num-train",num_train)

    if shuffle:
        # np.random.seed(0)
        np.random.shuffle(indices)
        # torch.manual_seed(0)

    train_idx, valid_idx = indices[split:], indices[:split]
    if test_as_valid:
        train_sampler = SubsetRandomSampler(indices)
    else:
        train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    if test_as_valid:

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
    else:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    # TODO visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        # plot_images(X, labels)

    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader
    }
    if test_as_valid:
        dataset_sizes = {"train": len(train_sampler),
                         "val": len(valid_dataset)}

    else:
        dataset_sizes = {"train": len(train_sampler),
                         "val": len(valid_sampler)}

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
            #transforms.Resize((output_width, output_height)),
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
            normalize,
        ])

        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader













def get_train_valid_loader_imagenet(batch_size,
                                    augment=True,
                                    data_dir = "./data/",
                                    output_width=224,
                                    output_height=224,
                                    valid_size=0.1,
                                    shuffle=True,
                                    show_sample=False,
                                    num_workers=16,
                                    pin_memory=True,
                                    test_as_valid =True):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg



    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # define transforms
        valid_transform =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


    # load the dataset

    train_dataset = datasets.ImageNet(
            root=data_dir, train=True,
            download=True, transform=train_transform,
    )

    if test_as_valid:

        valid_dataset = datasets.ImageNet(
            root=data_dir, train=False,
            download=True, transform=valid_transform,
        )

    else:
         valid_dataset = datasets.ImageNet(
                root=data_dir, train=True,
                download=True, transform=valid_transform,
        )


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    # print("spit ===> ",split,"num-train",num_train)

    if shuffle:
        # np.random.seed(0)
        np.random.shuffle(indices)
        # torch.manual_seed(0)

    train_idx, valid_idx = indices[split:], indices[:split]
    if test_as_valid:
        train_sampler = SubsetRandomSampler(indices)
    else:
        train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    if test_as_valid:

        valid_loader = get_test_loader_cifar(batch_size=batch_size,
                                             dataset="cifar100",
                                             output_width=output_width,
                                             output_height=output_height)
    else:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    # TODO visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        # plot_images(X, labels)

    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader
    }
    if test_as_valid:
        dataset_sizes = {"train": len(train_sampler),
                         "val": len(valid_dataset)}

    else:
        dataset_sizes = {"train": len(train_sampler),
                         "val": len(valid_sampler)}

    return data_loader_dict, dataset_sizes




def get_test_loader_imagenet(
                    batch_size,
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # define transform
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = datasets.ImageNet(
        root=data_dir, train=False,
        download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader










def get_train_valid_loader_svhn(batch_size,
                                    data_dir = "./data/",
                                    extra_training = True,
                                    output_width=32,
                                    output_height=32,
                                    shuffle=True,
                                    num_workers=16,
                                    pin_memory=True,
                                    ):
    #svhn_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #svhn_transform = transforms.Compose([
      #      transforms.ToTensor(),
      #  ])
    svhn_normalize = transforms.Normalize((0.430, 0.430, 0.446),(0.196, 0.198, 0.199))

    svhn_transform = transforms.Compose([
            transforms.ToTensor,
            svhn_normalize,

           #transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                         #std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    ])



    train_dataset = datasets.SVHN(
            root=data_dir, split="train",
            download=True, transform=svhn_transform,
    )
    if extra_training:
        train_dataset +=  datasets.SVHN(
            root=data_dir, split="extra",
            download=True, transform=svhn_transform,
        )



    valid_dataset = datasets.SVHN(
        root=data_dir, split="test",
        download=True, transform=svhn_transform,)


    num_train = len(train_dataset)
    indices = list(range(num_train))
    # print("spit ===> ",split,"num-train",num_train)

    if shuffle:
        # np.random.seed(0)
        np.random.shuffle(indices)
        # torch.manual_seed(0)


    train_sampler = SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = get_test_loader_cifar(batch_size=batch_size,
                                         output_width=output_width,
                                         output_height=output_height)

    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader
    }

    dataset_sizes = {"train": len(train_sampler),
                         "val": len(valid_dataset)}


    return data_loader_dict, dataset_sizes








def get_test_loader_svhn(
                    batch_size,
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
    svhn_normalize = transforms.Normalize((0.430, 0.430, 0.446),
                                          (0.196, 0.198, 0.199))

    transform = transforms.Compose([
        transforms.ToTensor,
        svhn_normalize,
        #transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                             #std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    ])


    #transform = transforms.Compose([
     #       transforms.ToTensor(),

      #  ])

    dataset = datasets.SVHN(
        root=data_dir, split="test",
        download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader




def load_tiny(data_folder="/home/aasadian/tiny_imagenet_200/", batch_size=64, phase='train',shuffle=True):


    #MEAN == > tensor([0.4802, 0.4481, 0.3975])
    #STD == > tensor([0.2770, 0.2691, 0.2821])

    transform_dict = {
        'train': transforms.Compose(
            [#transforms.Resize(256),
             #transforms.RandomCrop(224),
             #transforms.RandomHorizontalFlip(),
             #transforms.RandomHorizontalFlip(),
             #transforms.RandomCrop(64, padding=4),
             transforms.ToTensor(),
             #transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 # std=[0.2770, 0.2691, 0.2821]),
             ]),
        'test': transforms.Compose(
            [#transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2770, 0.2691, 0.2821]),
             ])}

    if phase == 'train':
        data_train = datasets.ImageFolder(root=data_folder+"train/", transform=transform_dict[phase])
        data_val = datasets.ImageFolder(root=data_folder+"val/", transform=transform_dict[phase])

        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                                                   num_workers=4)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False,
                                                 num_workers=4)

        data_loader_dict = {
            "train": train_loader,
            "val": val_loader
        }

        dataset_sizes = {"train": len(train_loader),
                         "val": len(val_loader)}


        return data_loader_dict,dataset_sizes
    else:
        data_test = datasets.ImageFolder(root=data_folder+"test/", transform=transform_dict[phase])

        test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, drop_last=False,
                                                    num_workers=4)
        return test_loader


"""
data_loader_dict,dataset_sizes = load_tiny(batch_size=1,shuffle=False)

from general_utils import get_mean_std_ptr,get_mean_std

mean,std = get_mean_std(data_loader_dict["train"])
mean_prt,std_prt = get_mean_std_ptr(data_loader_dict["train"])
print("Mean  ===> ",mean,"\t STD ==> ",std)
print("mean_prt  ===> ",mean_prt,"\t std_prt ==> ",std_prt)



data_loader_dict,dataset_sizes = load_tiny(batch_size=4)
print("Train size ====> ", dataset_sizes["train"], "Test Size ===> ", dataset_sizes["val"])#from general_utils import get_mean_std

#mean,std = get_mean_std(train_loader)
#print("MEAN  ==> ",mean)
#print("STD  ==> ",std)

import torchvision

it = iter(data_loader_dict["train"])
image,label = it.next()
image,label = it.next()
print("Label ===> ",(label.data))
import matplotlib.pyplot as plt
#image_2,label_2 = it.next()
def imshow(img):
 # img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()
imshow(torchvision.utils.make_grid(image))
#plt.imshow(image_2, interpolation='nearest')
#torchvision.utils.save_image(image_2,"/home/aasadian/ship.png")
#plt.savefig("/home/aasadian/plane.png")
#image,label = image.to("cuda:0"),label.to("cuda:0")
#print("Image type ",type(image))
#print("Label ===> ",load_label_names()[(int(label_2.data))])
"""
#get_train_valid_loader_cifars(batch_size=32)
