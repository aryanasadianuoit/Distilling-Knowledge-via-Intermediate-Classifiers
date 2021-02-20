import time
import copy
from torch import nn
from tqdm import tqdm
from dataload import *
from KD_Loss import kd_loss



def train_via_dih(student,
                      trained_core_teacher,
                      traind_intermediate_classifers_dict,
                      optimizer,
                      path_to_save,
                      seed,
                      dataset="cifar10",
                      epochs=200,
                      device_to_train_on="cuda",
                      multiple_gpu=None,
                      scheduler=None,
                      input_sample_size=(64,32,32),
                      kd_alpha=0.1,
                      kd_temperature=5):
    """
    :param student: un-trained student model
    :param trained_core_teacher: the main pre-trained teacher model
    :param traind_intermediate_classifers_dict:  a dictionary of mounted trained intermediate classifier heads
    :param optimizer: Default => SGD( lr: 0.1,Nesterov, momentum=0.9, with a scheduler)
    :param path_to_save:
    :param seed: seed for reproducibility
    :param dataset: default => CIFAR-100
    :param epochs: Default =>  200
    :param train_on: Default => cuda
    :param multiple_gpu: for check whether multi gpus are available. Deafult is None. This could be a list of GPU ids, when the
    device_to_train_on is set to cuda, and multiple gpus are available.
    :param scheduler: for changing the learning time through the training.
    :param input_sample_size:
    :param kd_alpha: dictionary/scalar containing alpha weights for distillation from each teacher(either the main teacher or each on intermediate classifiers)
    :param kd_temperature: dictionary/scalar containing alpha weights for distillation from each teacher(either the main teacher or each on intermediate classifiers)
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device_to_train_on:
        torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print("REPRODUCIBLE")


    device = torch.device(device_to_train_on)
    student.to(device)
    trained_core_teacher.to(device)
    trained_core_teacher.eval()
    for (intermediate_head_key, intermediate_classifer) in traind_intermediate_classifers_dict.items():
        intermediate_classifer.to(device)
        intermediate_classifer.eval()

    paralleled_intermediate_classifer_dict = {}
    if "cuda" in device_to_train_on:
        if multiple_gpu is not None:
            student = nn.DataParallel(student, device_ids=multiple_gpu)
            trained_core_teacher = nn.DataParallel(trained_core_teacher, device_ids=multiple_gpu)
            for (intermediate_head_key,intermediate_classifer) in traind_intermediate_classifers_dict.items():
                paralleled_intermediate_classifer_dict[intermediate_head_key] = nn.DataParallel(intermediate_classifer, device_ids=multiple_gpu)
                paralleled_intermediate_classifer_dict[intermediate_head_key].eval()
        else:
            print("MultiGPU is not active")
            paralleled_intermediate_classifer_dict = traind_intermediate_classifers_dict

    since = time.time()
    student.to(device)

    data_loader_dict, dataset_sizes = get_cifar(cifar10_100=dataset,
                                                batch_size=input_sample_size[0])

    test_loader = get_test_loader_cifar(dataset=dataset,
                                        batch_size=input_sample_size[0])

    best_model_wts = copy.deepcopy(student.state_dict())
    previous_saved_loss = 0.0
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
                    teacher_logits = trained_core_teacher(inputs)
                    if isinstance(student_outputs,tuple):
                        out_s = student_outputs[0]
                        _, preds = torch.max(student_outputs[0], 1)
                    else:
                        out_s = student_outputs
                        _, preds = torch.max(student_outputs, 1)


                    teacher_outputs = teacher_logits
                    final_teacher_logits = teacher_logits[0]
                    out_d_dict = {}

                    complete_teacher_outputs_dict = {}
                    complete_teacher_outputs_dict[0] = final_teacher_logits

                    for (intermediate_head_key,intermediate_classifer) in paralleled_intermediate_classifer_dict.items():
                        #for Non VGG Models
                        if not isinstance(teacher_outputs[1],list):
                            out_d_dict[intermediate_head_key] = intermediate_classifer(teacher_outputs[intermediate_head_key])
                            complete_teacher_outputs_dict[intermediate_head_key] =  intermediate_classifer(teacher_outputs[intermediate_head_key])
                        else:
                            out_d_dict[intermediate_head_key] = intermediate_classifer(teacher_outputs[1][intermediate_head_key-1])
                            complete_teacher_outputs_dict[intermediate_head_key] = intermediate_classifer(teacher_outputs[1][intermediate_head_key-1])


                    loss = 0.0
                    #if isinstance(final_gamma,dict):
                    for (branch_id,_) in complete_teacher_outputs_dict.items():
                        if isinstance(kd_alpha,dict):
                            alpha_value = kd_alpha[branch_id]
                        else:
                            alpha_value = kd_alpha

                        if alpha_value != 0.0:
                            each_header_loss = kd_loss(out_s=out_s,
                                                out_t=complete_teacher_outputs_dict[branch_id],
                                                target=labels,
                                                alpha=alpha_value,
                                                temperature=kd_temperature)
                            loss += each_header_loss

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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss <= previous_saved_loss:
                previous_saved_loss = epoch_loss
                print("Improvement in Val loss, saved!")
                torch.save(student.state_dict(), path_to_save)
                best_model_wts = copy.deepcopy(student.state_dict())
            elif phase == "val" and epoch_loss > previous_saved_loss:
                print("Previous saved loss is smaller! NOT saving.")


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))



    #EVALUATION


    # load best model weights
    student.load_state_dict(best_model_wts)
    since = time.time()
    student.eval()
    print("Test Evaluation In Process.... In ==> ", device)
    correct = 0
    total = 0
    with torch.no_grad():
        counter = 1
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device_to_train_on), labels.to(device_to_train_on)
            outputs = student(images)

            if isinstance(outputs, tuple):
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = (100 * (float(correct) / total))
            counter += 1

        time_elapsed = time.time() - since
        print('Evaluation Completed!')
        evaluation_log = "Test Acc ==> %.2f " % test_acc + "\t\t" + 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(evaluation_log)


