from tqdm import tqdm
import time
import copy
from dataload import get_cifar,get_test_loader_cifar
from KD_Loss import kd_loss
import numpy as np
import torch
from torch import nn
from DML_Loss import dml_loss_function

criterion = nn.CrossEntropyLoss()
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
        model = nn.DataParallel(model,device_ids=multiple_gpu)
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












#from defined_losses import *
#from tqdm import tqdm
#import copy
#import time


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

    device = torch.device(train_on)
    paralledled_middle_logits_model_dict = {}
    if ("cuda" in train_on) and torch.cuda.is_available():
        model = nn.DataParallel(model,device_ids=multiple_gpu)
        for key,middle_model in middle_logits_model_dict.items():
            if multiple_gpu is not None:
                paralledled_middle_logits_model_dict[key] = nn.DataParallel(middle_model,device_ids=multiple_gpu)
            else:
                paralledled_middle_logits_model_dict[key] = middle_logits_model_dict[key]
            paralledled_middle_logits_model_dict[key].to(device)
    since = time.time()
    model.to(device)

    data_loader_dict, dataset_sizes = get_cifar(cifar10_100=dataset,
                                                batch_size=batch_size)


    best_acc_branch_dict = {}
    best_model_save_path_dict = {}
    previous_loss_branch_dict = {}
    best_val_acc_branch_dict = {}
    best_train_acc_branch_dict = {}
    for (branch_key,branch) in paralledled_middle_logits_model_dict.items():
        best_acc_branch_dict[branch_key] = 0.0
        previous_loss_branch_dict[branch_key] = 0.0
        best_val_acc_branch_dict[branch_key] = 0.0
        best_train_acc_branch_dict[branch_key] = 0.0



    model.eval()

    for epoch in tqdm(range(epochs)):
        print('Epoch {}/{}'.format(epoch+1, epochs ))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train","val"]:
            if phase == 'train':
                for branch in paralledled_middle_logits_model_dict.values():
                    branch.train()

            else:
                for branch in paralledled_middle_logits_model_dict.values():
                    branch.eval()


            running_loss_middle_dict={}
            running_corrects_middle_dict = {}
            for branch_key,branch in paralledled_middle_logits_model_dict.items():
                running_loss_middle_dict[branch_key] = 0.0
                running_corrects_middle_dict[branch_key] = 0


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

                    branch_loss_dict ={}

                    total_loss = 0.0

                    for branch_key,branch in paralledled_middle_logits_model_dict.items():
                        #for VGG Models which returns the intermediate outputs in a list
                        if not isinstance(model_outputs[1],list):
                            branch_probs_out = branch(model_outputs[branch_key].detach())
                        else:
                            branch_probs_out = branch(model_outputs[1][branch_key-1].detach())

                        _, preds_dict[branch_key] = torch.max(branch_probs_out, 1)
                        branch_loss_dict[branch_key] = criterion(branch_probs_out,labels)
                        total_loss += branch_loss_dict[branch_key]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                for (key_branch,branch_predict) in preds_dict.items():
                    running_corrects_middle_dict[key_branch] += torch.sum(branch_predict == labels.data)


            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss_dict = {}
            for (key_branch, branch_loss_value) in branch_loss_dict.items():
                epoch_loss_dict[key_branch] = branch_loss_value / dataset_sizes[phase]

            epoch_acc_dict = {}
            for key, value in running_corrects_middle_dict.items():
                epoch_acc_dict[key] = value * 1.0 / dataset_sizes[phase]

            if phase == "val":
                for (branch_key,branch_previous_loss) in previous_loss_branch_dict.items():
                    if branch_previous_loss == 0.0:
                        previous_loss_branch_dict[branch_key]  =epoch_loss_dict[branch_key]

                for (branch_key,branch_best_val_acc) in best_val_acc_branch_dict.items():
                    if branch_best_val_acc == 0.0:
                        best_val_acc_branch_dict[branch_key]  =epoch_acc_dict[branch_key]

            if phase == "train":
                for (branch_key, branch_best_train_acc) in best_train_acc_branch_dict.items():
                    if branch_best_train_acc == 0.0:
                        best_train_acc_branch_dict[branch_key] = epoch_acc_dict[branch_key]


            for key in epoch_acc_dict.keys():
                print(phase,"\t",key,"\tACC: ",epoch_acc_dict[key].data,"\tLoss: ",epoch_loss_dict[key].data)


            if phase == "val":
                for (branch_key, branch_epoch_acc) in epoch_acc_dict.items():
                    if branch_epoch_acc > best_val_acc_branch_dict[branch_key]:
                        best_val_acc_branch_dict[branch_key] = epoch_acc_dict[branch_key]

            # deep copy the model

            if phase == 'val':
                for (branch_key,branch_previous_loss) in previous_loss_branch_dict.items():
                    if branch_previous_loss >= epoch_loss_dict[branch_key]:
                        previous_loss_branch_dict[branch_key] = epoch_loss_dict[branch_key]
                        best_val_acc_branch_dict[branch_key] = epoch_acc_dict[branch_key]


                        torch.save(paralledled_middle_logits_model_dict[branch_key].state_dict(),path_to_save+"_"+str(branch_key))


                       # _= checkpoint_saver(server=server,
                                           # experiment_name=experiment_name,
                                           # model_name=specific_file_name+str(branch_key),
                                           # epoch=epoch,
                                          #  model_state_dict=paralledled_middle_logits_model_dict[branch_key].state_dict(),
                                         #   optimizer_state_dict=optimizer.state_dict(),
                                         #   loss=loss,
                                        #    is_best=True)


                       # best_model_save_path_dict[branch_key] = experiment_result_saver(model=paralledled_middle_logits_model_dict[branch_key],
                                                                     # experiment_name=experiment_name,
                                                                     # server=server,
                                                                     # specific_file_name=specific_file_name + str(branch_key))

                    else:
                        print("Previous Validation Loss is smaller! Branch ==> ",branch_key)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))







def train_kd_or_fitnets_2(student,
                       trained_teacher,
                       optimizer,
                       path_to_save,
                       dataset="cifar10",
                       epochs =200,
                       train_on="cuda",
                       multiple_gpu=[0,1,2,3],
                       scheduler= None,
                       input_data_size=(64,32,32),
                       kd_alpha = 0.1,
                       kd_temperature = 5,
                       seed = 3
                       ):

    """

    :param student: either full student for KD or partial model up to guided layer. It can be Regressor added as well.
    :param trained_teacher: either full teacher or teacher up to hint layer.
    :param optimizer
    :param path_to_save: saving path for the trained model.
    :param dataset: Default :  CIFAR10
    :param epochs: Default : 200
    :param train_on: the main environment for training(CPU/GPU). Default : Cuda:0 .
    :param multiple_gpu: List of GPUs for data-parallelization
    :param scheduler:
    :param data_output_width: 32(CIFAR10/100)
    :param data_output_height: 32(CIFAR10/100)
    :param batch_size: 64
    :param kd_alpha: Default :0.1 NOTE ==> For FitNets stages, set it to None.
    :param kd_temperature: Default :5 NOTE ==> For FitNets stages, set it to None.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in train_on:
        torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print("REPRODUCIBLE")

    device = torch.device(train_on)
    student.to(device)
    trained_teacher.to(device)
    if "cuda" in train_on:
        student = nn.DataParallel(student, device_ids=multiple_gpu)
        trained_teacher = nn.DataParallel(trained_teacher, device_ids=multiple_gpu)
        trained_teacher.eval()
    since = time.time()
    student.to(device)
    trained_teacher.to(device)

    previous_saved_loss = 0.0
    if dataset == "cifar10" or dataset=="cifar100":
        data_loader_dict, dataset_sizes = get_cifar(batch_size=input_data_size[0],cifar10_100= dataset)

        test_loader = get_test_loader_cifar(dataset=dataset,
                                        batch_size=input_data_size[0],
                                        output_height=input_data_size[1],
                                        output_width=input_data_size[2])


    print("Train size ====> ",dataset_sizes["train"],"Val Size ===> ",dataset_sizes["val"])

    best_model_wts = copy.deepcopy(student.state_dict())
    train_acc_dict = {}
    train_loss_dict = {}
    val_acc_dict = {}
    val_loss_dict = {}

    for epoch in tqdm(range(epochs)):
        print('Epoch {}/{}'.format(epoch+1, epochs ))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                student.train()  # Set model to training mode
            else:
                student.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)


                # zero the parameter gradients
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    student_outputs = student(inputs)
                    teacher_logits = trained_teacher(inputs)
                    if isinstance(student_outputs,tuple):
                        out_s = student_outputs[0]
                        #_, preds = torch.max(student_outputs[0], 1)
                    else:
                        out_s = student_outputs

                    _, preds = torch.max(out_s, 1)

                    if isinstance(teacher_logits, tuple):
                        out_t = teacher_logits[0]
                    else:
                        out_t = teacher_logits

                    if kd_temperature != None and kd_alpha !=None:
                        #Knowledge Distillation.
                        # Criterion ==> L_total = (1-alph) * L_CE + (alpha * temperature **2) * D_KL( soft_student_output,soft teacher_output)
                        loss = kd_loss(out_s= out_s,
                                       out_t= out_t,
                                       target= labels,
                                       alpha=kd_alpha,
                                       temperature=kd_temperature)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects * 1.0 / dataset_sizes[phase]

            if phase == "train":
                train_acc_dict[epoch + 1] = epoch_acc
                train_loss_dict[epoch + 1] = epoch_loss
            elif phase == "val":
                val_acc_dict[epoch + 1] = epoch_acc
                val_loss_dict[epoch + 1] = epoch_loss
                if epoch == 0:
                    previous_saved_loss = epoch_loss

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss <= previous_saved_loss:
                previous_saved_loss = epoch_loss
                print("Improvement in Val loss, saved!")
                torch.save(student.state_dict(), path_to_save)
                best_model_wts = copy.deepcopy(student.state_dict())
            elif phase == "val" and epoch_loss > previous_saved_loss:
                print("Previous saved loss is smaller! NOT Saving.")


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    student.load_state_dict(best_model_wts)
    test_data_evaluation(student,
                         test_loader,
                         device=train_on,
                         state_dict=best_model_wts)



from tqdm import tqdm
import time
import copy

hint_loss_criterion = torch.nn.MSELoss()

def stage_1_fitnet_train(partial_student,
                       frozen_student_modules,
                       partail_teacher,
                       optimizer,
                       path_to_save,
                       guided_layer= None,
                       dataset="cifar10",
                       epochs =200,
                       train_on="cuda",
                       multiple_gpu=[0,1,2,3],
                       scheduler= None,
                       input_data_size=(64,32,32)):


    """

    :param partial_student: the student up to the guided layer/regressor.
    :param frozen_student_modules: the list of modules in the student that should not be trained, i.e., the layers after the guided/regressor layer.
    :param partail_teacher:
    :param optimizer : the trained and loaded teacher upto Hint layer.
    :param path_to_save
    :param guided_layer: if not None, it means that a regressor was needed due to dimension mismatch between the hint layer and th
     target intermediate layer in the student for the first stage of FitNets.
    :param dataset
    :param epochs: The total number of epcohs (default  40)
    :param train_on: The device to train on ("cuda"/"cpu")
    :param multiple_gpu: None
    :param scheduler: scheduler for adjusting the learning rate.
    :param input_data_size:
    :return: the trained parametrs upto guided layer in the student (state_dict)
    """


    device = torch.device(train_on)
    partial_student.to(device)
    partail_teacher.to(device)
    if "cuda" in train_on and multiple_gpu is not None:
        partial_student = nn.DataParallel(partial_student, device_ids=multiple_gpu)
        partail_teacher = nn.DataParallel(partail_teacher, device_ids=multiple_gpu)

    partail_teacher.eval()
    since = time.time()
    partial_student.to(device)
    partail_teacher.to(device)

    previous_saved_loss = 0.0
    data_loader_dict, dataset_sizes = get_cifar(batch_size=input_data_size[0],
                                                cifar10_100=dataset)



    print("Train size ====> ",dataset_sizes["train"],"Val Size ===> ",dataset_sizes["val"])

    best_model_wts = copy.deepcopy(partial_student.state_dict())
    train_loss_dict = {}
    val_loss_dict = {}

    for epoch in tqdm(range(epochs)):
        print('Epoch {}/{}'.format(epoch+1, epochs ))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                partial_student.train()  # Set model to training mode
                if guided_layer is not None:   # if due to dimension mismatch a regressor has been added, train the regressor too.
                    guided_layer.train()
                for m in frozen_student_modules:  # the layers after the guided layer or the added regressor should be frozen.
                    m.eval()

            else:
                partial_student.eval()   # Set model to evaluate mode
                if guided_layer is not None:
                    guided_layer.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loader_dict[phase]:
                inputs = inputs.to(device)
                #labels = labels.to(device)


                # zero the parameter gradients
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    student_outputs = partial_student(inputs)
                    teacher_logits = partail_teacher(inputs)
                    if isinstance(student_outputs,tuple):
                        out_s = student_outputs[1]
                    else:
                        out_s = student_outputs


                    if guided_layer is not None:
                        out_s = guided_layer(out_s)


                    if isinstance(teacher_logits, tuple):
                        out_t = teacher_logits[1]

                    loss = hint_loss_criterion(out_s,out_t)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)


            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]


            if phase == "train":
                train_loss_dict[epoch + 1] = epoch_loss
            elif phase == "val":
                val_loss_dict[epoch + 1] = epoch_loss
                if epoch == 0:
                    previous_saved_loss = epoch_loss

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss ))

            # deep copy the model
            if phase == 'val' and epoch_loss <= previous_saved_loss:
                previous_saved_loss = epoch_loss
                print("Improvement in Val loss, saved!")
                torch.save(partial_student.state_dict(), path_to_save)
                best_model_wts = copy.deepcopy(partial_student.state_dict())
            elif phase == "val" and epoch_loss > previous_saved_loss:
                print("Previous saved loss is smaller! NOT Saving.")


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return partial_student.state_dict()












def dml_train_regular(peers,
                      optimizers,
                      path_directory_to_save,
                      epochs=200,
                      train_on="cuda",
                      multiple_gpu=None,
                      dataset="cifar10",
                      scheduler=None,
                      alpha_dict={1:0.1,2:0.1},
                      temperature_dict=None,
                      data_input_size=(64, 32, 32),
                      seed=3):
    """
    :param peers: dictionary of peers {1:peer1,2:peer2,....}
    :param optimizers dictionary of optimizers {1:optimizer_peer1,2:optimizer_peer2,....}
    :param path_directory_to_save (it should be directory to save each peer by using its key value)
    :param epochs: deafult = 200
    :param train_on: cuda:cpu
    :param multiple_gpu: None
    :param dataset
    :param scheduler: dictionary of schedulers {1:scheduler_peer1,2:scheduler_peer2,....}
    :param alpha_dict: alpha for each peer. deafult = 0.1
    :param temperature_dict: tempearture for each peer ,default is 1
    :param data_input_size: default (62,32,32)
    :param seed: for reproducability default =3

    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in train_on:
        torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    peers_encyclopedia = {}
    paralleld_dict = {}

    device = torch.device(train_on)

    for (peer_key, peer) in peers.items():
        peer.to(device)
        if multiple_gpu is not None:
            peer = nn.DataParallel(peer, multiple_gpu)
        paralleld_dict[peer_key] = peer


    since = time.time()
    data_loader_dict, dataset_sizes = get_cifar(batch_size=data_input_size[0],cifar10_100=dataset)


    for (key, value) in paralleld_dict.items():
        peers_encyclopedia["best_model_wts_" + key] = copy.deepcopy(peers[key].state_dict())
        peers_encyclopedia["best_train_acc_" + key] = 0.0
        peers_encyclopedia["best_train_loss_" + key] = 0.0
        peers_encyclopedia["best_val_acc_" + key] = 0.0
        peers_encyclopedia["best_val_loss_" + key] = 0.0
        peers_encyclopedia["best_acc_dict_" + key] = {}
        peers_encyclopedia["running_loss_dict_" + key] = {}
        peers_encyclopedia["running_corrects_dict_" + key] = {}
        peers_encyclopedia["train_loss_dict_" + key] = {}
        peers_encyclopedia["train_acc_dict_" + key] = {}
        peers_encyclopedia["val_acc_dict_" + key] = {}
        peers_encyclopedia["val_loss_dict_" + key] = {}
        peers_encyclopedia["previous_loss_" + key] = 0.0
        peers_encyclopedia["periodic_saved_checkpoint_val_loss_" + key] = {}
        peers_encyclopedia["best_saved_checkpoint_val_loss_" + key] = 0.0

    for epoch in tqdm(range(epochs)):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for (key, peer) in paralleld_dict.items():
            print("\t\t \t \t KEY :::::::::::::::::::::>   ", key)


            final_alpha = alpha_dict[key]

            for phase in ['train', 'val']:
                optimizer = optimizers[key]

                if phase == 'train':
                    peer.train()  # Set model to training mode
                else:
                    peer.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for (inputs, labels) in data_loader_dict[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        optimizer.zero_grad()

                        student_input = inputs
                        peer_final_output = peer(student_input)

                        if isinstance(peer_final_output, tuple):
                            peer_final_output = peer_final_output[0]

                        teachers_final_ouput_dict = {}
                        for key_others, other_peer in paralleld_dict.items():
                            if key_others != key:
                                other_peer_final_output = other_peer(inputs)
                                if isinstance(other_peer_final_output, tuple):
                                    teachers_final_ouput_dict[key_others] = other_peer_final_output[0]
                                else:
                                    teachers_final_ouput_dict[key_others] = other_peer_final_output

                        # Check whether a regressor has been added in the forward method and the output
                        # is a tuple or the model(forward's method has not been modified, so the output of the model is a tensor)
                        if isinstance(peer_final_output, tuple):
                            _, preds = torch.max(peer_final_output[0], 1)
                        else:
                            _, preds = torch.max(peer_final_output, 1)

                        if temperature_dict != None:
                            if temperature_dict[key] != None:
                                final_temperature = temperature_dict[key]
                            else:
                                final_temperature = 1
                        else:
                            final_temperature = 1


                        loss = dml_loss_function(student_peer_output=peer_final_output,
                                                 teacher_peers_outputs=teachers_final_ouput_dict,
                                                 target_labels=labels,
                                                 alpha=final_alpha,
                                                 temperature=final_temperature)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # loss.backward()
                            torch.autograd.backward(loss)
                            optimizer.step()

                    input_batch_size = inputs.size(0)
                    running_loss += loss * input_batch_size
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and scheduler != None:
                    if scheduler[key]:
                        scheduler[key].step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == "val" and epoch == 0:
                    peers_encyclopedia["previous_loss_" + key] = epoch_loss
                    peers_encyclopedia["best_val_acc_" + key] = epoch_acc

                if phase == "train" and epoch == 0:
                    peers_encyclopedia["best_train_acc_" + key] = epoch_acc

                print('Student => {} ===> {} Loss: {:.4f} Acc: {:.4f}'.format(key,
                                                                              phase, epoch_loss, epoch_acc))
                if phase == "train":
                    peers_encyclopedia["train_acc_dict_" + key][epoch + 1] = epoch_acc
                    peers_encyclopedia["train_loss_dict_" + key][epoch + 1] = epoch_loss

                elif phase == "val":
                    peers_encyclopedia["val_acc_dict_" + key][epoch + 1] = epoch_acc
                    peers_encyclopedia["val_loss_dict_" + key][epoch + 1] = epoch_loss
                    if epoch_acc > peers_encyclopedia["best_val_acc_" + key]:
                        peers_encyclopedia["best_val_acc_" + key] = epoch_acc
                        peers_encyclopedia["best_acc_dict_" + key] = epoch_acc
                if phase == "val" and peers_encyclopedia["previous_loss_" + key] >= epoch_loss:
                    peers_encyclopedia["previous_loss_" + key] = epoch_loss
                    torch.save(peer.state_dict,path_directory_to_save+"/"+key+".pth")
                    peers_encyclopedia["best_model_wts_" + key] = copy.deepcopy(peer.state_dict())

                    peers_encyclopedia["best_saved_checkpoint_val_loss_" + key] = epoch_loss
                    print("Improvement in VAL Loss ==> SAVED!  EPOCH ====>", str(epoch + 1))
                if phase == "val" and peers_encyclopedia["previous_loss_" + key] < epoch_loss:
                    print("Previous Validation Loss is smaller!")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

























