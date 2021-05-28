from tqdm import tqdm
import time
import torch
import copy
from torch import nn
from general_utils import experiment_result_saver
from general_utils import test_data_evaluation
from general_utils import checkpoint_saver
from models.resnet_cifar import resnet110_cifar,resnet20_cifar
from general_utils import get_optimizer_scheduler
from torchvision import transforms
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


from general_utils import reproducible_state



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
            #transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
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
           # transforms.Resize((output_width, output_height)),
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













def train_grid_regular_ce(model,
                  optimizer,
                  experiment_name,
                  criterion = nn.CrossEntropyLoss(),
                  dataset="cifar100",
                  epochs =200,
                  train_on="cuda:3",
                  multiple_gpu=None,
                  scheduler= None,
                  input_data_size = (128,32,32),
                  checkpoint_saved_every_x_epochs = 5):

    device = torch.device(train_on)
    if "cuda" in train_on and multiple_gpu != None:
        model = nn.DataParallel(model,device_ids=multiple_gpu)
    since = time.time()
    model.to(device)

    if dataset == "cifar10" or dataset=="cifar100":

        data_loader_dict,dataset_sizes = get_train_valid_loader_cifars(batch_size=input_data_size[0],
                                                                       cifar10_100=dataset,)
        test_loader = get_test_loader_cifar(batch_size=input_data_size[0],
                                            dataset=dataset,)


    best_model_wts = copy.deepcopy(model.state_dict())
    previous_loss = 0.0
    best_val_acc = 0.0
    best_train_acc = 0.0

    train_acc_dict = {}
    train_loss_dict = {}
    val_acc_dict = {}
    val_loss_dict = {}

    #for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        #print('Epoch {}/{}'.format(epoch+1, epochs ))
        #print('-' * 10)


        for phase in ["train","val"]: #phase_list:#['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    model_outputs = model(inputs)
                    loss = 0.0
                    losses = {}


                    #loss = torch.tensor(0.0,requires_grad=True,device=device)
                    #loss = Variable(0.0, requires_grad=True)

                    for head_index in range(len(model_outputs[0])):


                        head_out = model_outputs[0][head_index]
                        #if head_index == 0:
                            #head_out = model_outputs[0][head_index]#.detach()

                        #else:
                            #head_out = model_outputs[0][head_index]#.detach()
                        #_, preds = torch.max(head_out, 1)
                        #losses[head_index] = criterion(head_out, labels)
                        losses[head_index] = criterion(head_out, labels)

                        if head_index == 0:
                            _, preds = torch.max(head_out, 1)
                        #if head_index == 0:
                          #  loss_0 = criterion(head_out, labels)
                        #elif head_index == 1:
                           # loss_1 = criterion(head_out, labels)

                        #elif head_index == 2:
                          #  loss_2 = criterion(head_out, labels)

                        #loss += criterion(head_out, labels)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #total_loss = Variable(loss, requires_grad=True)
                        total_loss = sum(losses.values())
                        #total_loss = loss_0+loss_1 +loss_2
                        total_loss.backward()
                        #total_loss.backward(retain_graph= True)
                        #total_loss =sum(losses.values())
                        #torch.autograd.backward([loss_0,loss_1,loss_2],retain_graph=True)
                        #total_loss.backward()
                        optimizer.step()

                #running_loss += loss.item() * inputs.size(0)
                #running_loss += loss.item() * inputs.size(0)
                running_loss += total_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            if previous_loss == 0.0  and phase == "val":
                previous_loss = epoch_loss

            epoch_acc = running_corrects * 1.0 / dataset_sizes[phase]
            if best_val_acc == 0.0 and phase == "val":
                best_val_acc = epoch_acc
            if best_train_acc == 0.0  and phase == "train":
                best_train_acc = epoch_acc

            #print('{} Loss: {:.4f}  ACC: {:.4f}'.format(
               # phase, epoch_loss,epoch_acc))
            if phase == "train":
                train_acc_dict[(epochs + 1 )] = epoch_acc
                train_loss_dict[(epoch + 1 )] = epoch_loss
            elif phase == "val":
                val_acc_dict[(epoch + 1 )] = epoch_acc
                val_loss_dict[(epoch + 1 )] = epoch_loss

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc

            # deep copy the model
            if phase == 'val' and previous_loss >= epoch_loss:
                previous_loss = epoch_loss

                best_acc = epoch_acc
                #best_model_checkpoint_path = checkpoint_saver(server=server,
                                                              #experiment_name=experiment_name,
                                                              #model_name=specific_file_name,
                                                              #epoch=epoch,
                                                              #model_state_dict=model.state_dict(),
                                                              #optimizer_state_dict=optimizer.state_dict(),
                                                              #loss=loss,
                                                              #add_path_to_root=add_path_to_root,
                                                              #is_best=True)
                torch.save(model.state_dict(),"/home/aasadian/tofd/teacher/"+experiment_name+".pth")
                #saved_path = experiment_result_saver(model=model,
                                                     #experiment_name=experiment_name,
                                                     #server= server,
                                                     #specific_file_name= specific_file_name,
                                                     #add_path_to_root=add_path_to_root)
                best_model_wts = copy.deepcopy(model.state_dict())
            #elif phase == "val"  and previous_loss < epoch_loss:
                #print("Previous Validation Loss is smaller!")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)





    since = time.time()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        counter = 1
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                # TODO change outputs[1] to outputs[0]
                _, predicted = torch.max(outputs[0][0].data, 1)
            else:
                _, predicted = torch.max(outputs[0][0].data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_acc = (100 * (float(correct) / total))
            counter += 1

        time_elapsed = time.time() - since
        print('Evaluation Completed!')
        evaluation_log = "Test Acc ==> %.2f " % test_acc + "\t\t" + 'Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(evaluation_log)


SEEDS = [50]
setting = "dih"
DEVICE = "cuda:3"


for SEED in SEEDS:
    reproducible_state(seed=SEED,device=DEVICE)
    from VGG_TOFD import VGG_Intermediate_Branches_TOFD

    teacher = VGG_Intermediate_Branches_TOFD("VGG11",seed=SEED,num_classes=100)


    if setting == "dih":
        EPOCHS =200
        optimizer,scheduler = get_optimizer_scheduler(teacher)

    else:
        EPOCHS = 250
        optimizer = torch.optim.SGD(teacher, lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 240], gamma=0.1,
                                                     last_epoch=-1)







    train_grid_regular_ce(teacher,
                      optimizer=optimizer,
                      experiment_name="teacher_vgg11_tofd_seed_"+str(SEED),
                      input_data_size=(128,32,32),
                      scheduler=scheduler,
                      epochs=EPOCHS,
                      train_on=DEVICE)