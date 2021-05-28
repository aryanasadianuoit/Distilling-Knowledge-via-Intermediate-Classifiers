from tqdm import tqdm
import time
import copy
from dataloader import *
from defined_losses import *
from general_utils import experiment_result_saver
from general_utils import test_data_evaluation
from losses.regular_kd_loss import kd_loss
from general_utils import checkpoint_saver
from general_utils import plotter
from losses.KD_Loss_Module import DistillKL
import math
from torch.utils.tensorboard import SummaryWriter
from tiny_data import Tiny

criterion = nn.CrossEntropyLoss()


def train_mhkd_grid(student,
                   trained_core_teacher,
                   teacher_headers_dict,
                   student_headers_dict,
                   optimizer,
                   path_to_save,
                   mhkd_beta = 0.5,
                   dataset="cifar100",
                   epochs =200,
                   train_on="cuda:3",
                   multiple_gpu=None,
                   scheduler= None,
                   input_sample_size = (128,32,32),
                   kd_alpha = 0.1,
                   kd_temperature = 5,
                   cosine_annealing_gamma = False,
                   server=2):

    """

    :param student_partially_trained_from_stage_1:
    :param trained_teacher:
    :param optimizer:
    :param path_to_save:
    :param dataset:
    :param epochs:
    :param train_on:
    :param multiple_gpu:
    :param scheduler:
    :param data_output_width:
    :param data_output_height:
    :param batch_size:
    :param kd_alpha:
    :param kd_temperature:
    :param no_vidaltion: If True, then the test set is used for both testing and Also Validation. If no, a split
    of train set is going to be used as validation set.
    :return:
    """



    device = torch.device(train_on)
    student.to(device)
    trained_core_teacher.to(device)
    trained_core_teacher.eval()
    for (branch_key, branch_classifer) in teacher_headers_dict.items():
        branch_classifer.to(device)

    for (student_head_key, student_head_value) in student_headers_dict.items():
        student_head_value.to(device)

    paralleled_teacher_headers_dict = {}
    if "cuda" in train_on:
        if multiple_gpu is not None:

            student = nn.parallel.DistributedDataParallel(student, device_ids=multiple_gpu)
            trained_core_teacher = nn.parallel.DistributedDataParallel(trained_core_teacher, device_ids=multiple_gpu,output_device=1)
            for (branch_key,branch_classifer) in teacher_headers_dict.items():
                paralleled_teacher_headers_dict[branch_key] = nn.DataParallel(branch_classifer, device_ids=multiple_gpu)
        else:

            paralleled_teacher_headers_dict = teacher_headers_dict

    since = time.time()
    student.to(device)
    #trained_teacher.to(device)





    previous_saved_loss = 0.0
    if dataset == "cifar10" or dataset == "cifar100":
        data_loader_dict, dataset_sizes = get_train_valid_loader_cifars(batch_size=input_sample_size[0],
                                                                        cifar10_100=dataset)

        test_loader = get_test_loader_cifar(batch_size=input_sample_size[1],
                                        dataset=dataset)
    elif dataset == "svhn":
        data_loader_dict, dataset_sizes = get_train_valid_loader_svhn(batch_size=input_sample_size[0],extra_training=True)

        test_loader = get_test_loader_svhn(batch_size=input_sample_size[0])
    elif dataset == "tiny":
        tiny = Tiny(batch_size=input_sample_size[0],server=server)
        data_loader_dict, dataset_sizes = tiny.data_loader_dict, tiny.dataset_sizes
        test_loader = tiny.data_loader_dict["val"]



    best_model_wts = copy.deepcopy(student.state_dict())
    best_acc = 0.0
    train_acc_dict = {}
    train_loss_dict = {}
    val_acc_dict = {}
    val_loss_dict = {}

    #for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):

        final_alpha = kd_alpha

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                student.train()  # Set model to training mode
                for added_teacher_head, added_student_head in zip(paralleled_teacher_headers_dict.values(),
                                                                  student_headers_dict.values()):
                    added_student_head.train()
                    added_teacher_head.train()
            else:
                student.eval()   # Set model to evaluate mode
                for added_teacher_head, added_student_head in zip(paralleled_teacher_headers_dict.values(),student_headers_dict.values()):
                    added_student_head.eval()
                    added_teacher_head.eval()

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
                    teacher_logits = trained_core_teacher(inputs)

                    complete_student_outputs_dict = {}

                    if isinstance(student_outputs,tuple):
                        #for student_head_index in range(len(student_outputs)):
                        complete_student_outputs_dict[0] = student_outputs[0]
                        out_s = student_outputs[0]

                        _, preds = torch.max(student_outputs[0], 1)
                    else:
                        out_s = student_outputs
                        _, preds = torch.max(student_outputs, 1)



                    for (student_head_key,student_head_value) in student_headers_dict.items():
                        #for Non VGG Models
                        if not isinstance(student_outputs,list):
                            complete_student_outputs_dict[student_head_key] =  student_head_value(student_outputs[student_head_key])
                        else:
                            complete_student_outputs_dict[student_head_key] = student_head_value(student_outputs[1][student_head_key-1])




                    teacher_outputs = teacher_logits
                    final_teacher_logits = teacher_logits[-1]
                    out_d_dict = {}

                    complete_teacher_outputs_dict = {}
                    complete_teacher_outputs_dict[0] = final_teacher_logits

                    for (teacher_head_key,teacher_head_value) in paralleled_teacher_headers_dict.items():

                        if isinstance(teacher_outputs,list):

                            out_d_dict[teacher_head_key] = teacher_head_value(teacher_outputs[teacher_head_key-1])
                            complete_teacher_outputs_dict[teacher_head_key] = teacher_head_value(
                                teacher_outputs[teacher_head_key-1])

                 
                    loss = 0.0
                    if isinstance(final_alpha,dict):
                        for (teacher_head_id,_),(student_head_id,_) in zip(complete_teacher_outputs_dict.items(),complete_student_outputs_dict.items()):

                            # teachers auxillary headers loss (regular cross-entropy)
                            if teacher_head_id != 0: # main classifier is full-trained, By added classifiers should be trained with cross-entropy
                                temp = criterion(complete_teacher_outputs_dict[teacher_head_id],labels)
                            else:
                                temp = 0.0

                            if final_alpha[teacher_head_id] != 0.0:
                                # KD between the intermediate heads
                                if teacher_head_id != 0 and student_head_id != 0:
                                    temp += mhkd_beta *kd_loss(out_s=complete_student_outputs_dict[student_head_id],
                                                   out_t=complete_teacher_outputs_dict[teacher_head_id],
                                                   target=labels,
                                                   gamma=final_alpha[teacher_head_id],
                                                   temperature=kd_temperature)
                                # KD between the final heads
                                elif teacher_head_id == 0 and student_head_id == 0:
                                    temp += kd_loss(out_s=complete_student_outputs_dict[student_head_id],
                                               out_t=complete_teacher_outputs_dict[student_head_id],
                                               target=labels,
                                               gamma=final_alpha[student_head_id+1],
                                               temperature=kd_temperature)
                                loss += temp

                    else:
                        for (teacher_head_id, _) in complete_teacher_outputs_dict.items():
                            #Todo like above
                            loss += kd_loss(out_s=out_s,
                                            out_t=complete_teacher_outputs_dict[teacher_head_id],
                                            target=labels,
                                            gamma=final_alpha,
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
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                train_acc_dict[epoch + 1] = epoch_acc
                train_loss_dict[epoch + 1] = epoch_loss
            elif phase == "val":
                val_acc_dict[epoch + 1] = epoch_acc
                val_loss_dict[epoch + 1] = epoch_loss
                if epoch == 0:
                    previous_saved_loss = epoch_loss

  
            # deep copy the model
            if phase == 'val' and epoch_loss <= previous_saved_loss:
                previous_saved_loss = epoch_loss
                #print("Improvement in Val loss, saved!")
                torch.save(student.state_dict(), path_to_save)
                best_model_wts = copy.deepcopy(student.state_dict())



    time_elapsed = time.time() - since

    # load best model weights
    student.load_state_dict(best_model_wts)
    _,test_acc = test_data_evaluation(student,
                         test_loader,
                         device=train_on,
                         state_dict=best_model_wts)

    return test_acc,time_elapsed

