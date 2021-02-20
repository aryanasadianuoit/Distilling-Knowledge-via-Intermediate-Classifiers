from tqdm import tqdm
import time
import copy
from dataload import get_cifar
import torch
#from defined_losses import *


criterion = torch.nn.CrossEntropyLoss()
def train_regular_ce(model,
                  optimizer,
                  path_to_save,
                  dataset="cifar10",
                  epochs = 200,
                  train_on="cuda:0",
                  multiple_gpu=None,
                  scheduler= None,
                  batch_size = 64):

    device = torch.device(train_on)
    if ("cuda" in train_on) and (multiple_gpu is not None):
        model = torch.nn.DataParallel(model,device_ids=multiple_gpu)
    since = time.time()
    model.to(device)


    data_loader_dict,dataset_sizes = get_cifar(batch_size=batch_size,
                                                   cifar10_100=dataset)


    best_model_wts = copy.deepcopy(model.state_dict())
    previous_loss = 0.0
    best_val_acc = 0.0
    best_train_acc = 0.0

    train_acc_dict = {}
    train_loss_dict = {}
    val_acc_dict = {}
    val_loss_dict = {}

    for epoch in tqdm(range(epochs)):
        print('Epoch {}/{}'.format(epoch+1, epochs ))
        print('-' * 10)


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
                    if isinstance(model_outputs,tuple):
                        _, preds = torch.max(model_outputs[0], 1)
                        loss = criterion(model_outputs[0], labels)
                    else:
                        _, preds = torch.max(model_outputs, 1)
                        loss = criterion(model_outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
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

            print('{} Loss: {:.4f}  ACC: {:.4f}'.format(
                phase, epoch_loss,epoch_acc))
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

                torch.save(model.state_dict(),path_to_save)
                best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == "val"  and previous_loss < epoch_loss:
                print("Previous Validation Loss is smaller!")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best VAL Acc: {:4f}'.format(best_val_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()
    return model






def train_regular_middle_logits(model,
                                optimizer,
                                path_to_save,
                                middle_logits_model_dict,
                                dataset="cifar10",
                                epochs =200,
                                train_on="cuda",
                                multiple_gpu=None,
                                scheduler= None,
                                batch_size = 64):

    """

    :param model: the core model which the intermediate heads should be mounted on.
    :param optimizer: optimizer
    :param path_to_save: path to save the fine_tuned intermediate heads. this should be a directory. The function will save the
    fine_tuned intermediate heads based on the index of the intermediate output from core mode that they have used as input. saved models example:
    [ _1.pth,_2.pth,_3.pth,...._n.pth]
    :param middle_logits_model_dict: dictionary containing the intermediate classifier models( just the classifier module)
    these modules should be stacked on some pre-defined intermediate layers of the core model
    :param dataset: dataset default = cifar100
    :param epochs: number of epochs default = 200
    :param train_on: either gou or cpu. default : cuda:0
    :param multiple_gpu: None, or in the case of using multiple gpus, list the ids ex: [0,1,2]
    :param scheduler: scheduler instance for adjusting the learning rate through the training process.
    :param batch_size: default = 64
    """

    device = torch.device(train_on)
    paralledled_middle_logits_model_dict = {}

    if ("cuda" in train_on) and torch.cuda.is_available():
        for key,middle_model in middle_logits_model_dict.items():
            if multiple_gpu is not None:
                paralledled_middle_logits_model_dict[key] = torch.nn.DataParallel(middle_model,device_ids=multiple_gpu)
            else:
                paralledled_middle_logits_model_dict[key] = middle_logits_model_dict[key]
            paralledled_middle_logits_model_dict[key].to(device)
    since = time.time()
    model.to(device)

    data_loader_dict, dataset_sizes = get_cifar(cifar10_100=dataset,
                                                batch_size=batch_size)

    head_loss_dict={}
    best_acc_head_dict = {}
    previous_loss_head_dict = {}
    best_val_acc_head_dict = {}
    best_train_acc_head_dict = {}
    previous_head_dict  ={}
    epoch_loss_dict = {}
    for (head_key,head) in paralledled_middle_logits_model_dict.items():
        best_acc_head_dict[head_key] = 0.0
        previous_head_dict[head_key] = 0.0
        best_val_acc_head_dict[head_key] = 0.0
        best_train_acc_head_dict[head_key] = 0.0



    model.eval()  #The core model is FROZEN during the fine_tuning

    for epoch in tqdm(range(epochs)):
        print('Epoch {}/{}'.format(epoch+1, epochs ))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train","val"]:
            if phase == 'train':
                for head in paralledled_middle_logits_model_dict.values():
                    head.train()

            else:
                for head in paralledled_middle_logits_model_dict.values():
                    head.eval()


            running_loss_middle_dict={}
            running_corrects_middle_dict = {}
            for head_key,head in paralledled_middle_logits_model_dict.items():
                running_loss_middle_dict[head_key] = 0.0
                running_corrects_middle_dict[head_key] = 0


            # Iterate over data.
            for inputs, labels in data_loader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    model_outputs = model(inputs)
                    preds_dict ={}
                    if isinstance(model_outputs,tuple):
                        _, preds = torch.max(model_outputs[0], 1)
                        #loss = criterion(model_outputs[0], labels)
                    else:
                        _, preds = torch.max(model_outputs, 1)
                        #loss = criterion(model_outputs,labels)

                    head_loss_dict= {}

                    total_loss = 0.0

                    for head_key,head in paralledled_middle_logits_model_dict.items():

                        # for Non VGG Models which return all the outputs in this order : final_output,internal_output_1,internal_output2, ....., internal_output_n
                        if not isinstance(model_outputs[1],list):
                            head_probs_out = head(model_outputs[head_key].detach())

                        # for VGG Models which return the intermediate outputs as a list
                        else:
                            head_probs_out = head(model_outputs[1][head_key-1].detach())

                        _, preds_dict[head_key] = torch.max(head_probs_out, 1)
                        head_loss_dict[head_key] = criterion(head_probs_out,labels)
                        total_loss += head_loss_dict[head_key]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                for (head_key,head_predict) in preds_dict.items():
                    running_corrects_middle_dict[head_key] += torch.sum(head_predict == labels.data)


            if phase == 'train' and scheduler != None:
                scheduler.step()

            #epoch_loss_dict = {}
            for (head_key, head_loss_value) in head_loss_dict.items():
                epoch_loss_dict[head_key] = head_loss_value / dataset_sizes[phase]

            epoch_acc_dict = {}
            for key, value in running_corrects_middle_dict.items():
                epoch_acc_dict[key] = value * 1.0 / dataset_sizes[phase]

            if phase == "val":
                if len(previous_loss_head_dict.keys()) == 0:
                    for (head_key, head_previous_loss) in epoch_loss_dict.items():
                        previous_loss_head_dict[head_key] = epoch_loss_dict[head_key]
                else:
                    for (head_key, head_previous_loss) in previous_loss_head_dict.items():
                        previous_loss_head_dict[head_key] = epoch_loss_dict[head_key]


                for (head_key,head_previous_loss) in previous_loss_head_dict.items():
                    if head_previous_loss == 0.0:
                        print("HERE")
                        previous_loss_head_dict[head_key] = epoch_loss_dict[head_key]


                for (head_key,head_best_val_acc) in best_val_acc_head_dict.items():
                    if head_best_val_acc == 0.0:
                        best_val_acc_head_dict[head_key]  =epoch_acc_dict[head_key]

            if phase == "train":
                for (head_key, head_best_train_acc) in best_train_acc_head_dict.items():
                    if head_best_train_acc == 0.0:
                        best_train_acc_head_dict[head_key] = epoch_acc_dict[head_key]


            if phase == "val":
                for (head_key, head_epoch_acc) in epoch_acc_dict.items():
                    if head_epoch_acc > best_val_acc_head_dict[head_key]:
                        best_val_acc_head_dict[head_key] = epoch_acc_dict[head_key]

            # deep copy the model

            if phase == 'val':   #check whether the validation loss is lower in this epoch or not. If yes, save the model.
                for (head_key,head_previous_loss) in previous_loss_head_dict.items():
                    if head_previous_loss >= epoch_loss_dict[head_key]:
                        previous_loss_head_dict[head_key] = epoch_loss_dict[head_key]
                        best_val_acc_head_dict[head_key] = epoch_acc_dict[head_key]
                        print("SAVE")

                        torch.save(paralledled_middle_logits_model_dict[head_key].state_dict(),path_to_save+"_"+str(head_key)+".pth")

                    else:
                        print("Previous Validation Loss is smaller! Branch ==> ",head_key)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))






















