from tqdm import tqdm
import time
import copy
from dataloader import *
from defined_losses import *
from general_utils import experiment_result_saver
from general_utils import test_data_evaluation
from general_utils import checkpoint_saver
from tiny_data import Tiny





def train_grid_regular_ce(model,
                  optimizer,
                  experiment_name,
                  server,
                  specific_file_name,
                  additional_note,
                  criterion = nn.CrossEntropyLoss(),
                  dataset="cifar100",
                  epochs =200,
                  train_on="cuda:1",
                  multiple_gpu=[1],
                  scheduler= None,
                  added_regressor = None,
                  input_data_size = (128,32,32),
                  add_path_to_root =None):

    device = torch.device(train_on)
    if "cuda" in train_on:
        model = nn.DataParallel(model,device_ids=multiple_gpu)
    since = time.time()
    model.to(device)

    if dataset == "cifar10" or dataset=="cifar100":

        data_loader_dict,dataset_sizes = get_train_valid_loader_cifars(batch_size=input_data_size[0],
                                                                   cifar10_100=dataset)
        test_loader = get_test_loader_cifar(batch_size=input_data_size[0],
                                        dataset=dataset,)
    else:

        tiny = Tiny(batch_size=input_data_size[0], server=server)
        # data_loader_dict,dataset_sizes =load_tiny(batch_size=input_data_size[0],phase="train")
        data_loader_dict, dataset_sizes = tiny.data_loader_dict, tiny.dataset_sizes
        test_loader = tiny.data_loader_dict["val"]



    #print("Train size ====> ", dataset_sizes["train"], "Test Size ===> ", dataset_sizes["val"])


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    previous_loss = 0.0
    best_val_acc = 0.0
    best_train_acc = 0.0

    train_acc_dict = {}
    train_loss_dict = {}
    val_acc_dict = {}
    val_loss_dict = {}

    #for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs ))
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
                    if isinstance(model_outputs,tuple):
                        _, preds = torch.max(model_outputs[0][0], 1)
                        loss = 0.0
                        for output_head in model_outputs[0]:
                            loss += criterion(output_head,labels)

                        #loss = criterion(model_outputs[0], labels)
                    else:
                        _, preds = torch.max(model_outputs, 1)
                        loss = criterion(model_outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if added_regressor == None:
                            loss.backward()
                        else:
                            #print("TODO_added_regressor is Not NONE!")
                            loss.backward()
                            #torch.autograd.backward()
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

                best_acc = epoch_acc
                best_model_checkpoint_path = checkpoint_saver(server=server,
                                                              experiment_name=experiment_name,
                                                              model_name=specific_file_name,
                                                              epoch=epoch,
                                                              model_state_dict=model.state_dict(),
                                                              optimizer_state_dict=optimizer.state_dict(),
                                                              loss=loss,
                                                              add_path_to_root=add_path_to_root,
                                                              is_best=True)
                saved_path = experiment_result_saver(model=model,
                                                     experiment_name=experiment_name,
                                                     server= server,
                                                     specific_file_name= specific_file_name,
                                                     add_path_to_root=add_path_to_root)
                best_model_wts = copy.deepcopy(model.state_dict())
            #elif phase == "val"  and previous_loss < epoch_loss:
                #print("Previous Validation Loss is smaller!")

    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(
        #time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc with Lowest val Loss: {:4f}'.format(best_acc))
    #print('Best VAL Acc: {:4f}'.format(best_val_acc))
    #additional_note += "\n"+ 'Best val Acc with Lowest val Loss: {:4f}'.format(best_acc) + \
    #"\n" + 'Best VAL Acc: {:4f}'.format(best_val_acc)+"\n"

    # load best model weights
    model.load_state_dict(best_model_wts)

    total_checkpoint = torch.load(best_model_checkpoint_path)
    best_state_dict = total_checkpoint["model_state_dict"]



    #if added_regressor != None:
        ##Todo
       # print("TODO for Added Regressor")


    #else:
    evaluation_log,test_acc_value = test_data_evaluation(model=model,
                                              test_loader=test_loader,
                                              state_dict=best_state_dict,
                                              saved_load_state_path=None,#saved_path,
                                              added_regressor = None,
                                              device=train_on,
                                              server=server)

    experiment_result_saver(model= model,
                                experiment_name=experiment_name,
                                server=server,
                                specific_file_name=specific_file_name,
                                additional_note=additional_note,
                                train_acc_dict = train_acc_dict,
                                train_loss_dict =train_loss_dict,
                                val_acc_dict =val_acc_dict,
                                val_loss_dict = val_loss_dict,
                                test_result_log= evaluation_log,
                            add_path_to_root= add_path_to_root)

    return  test_acc_value



from wres_2 import *
net =get_Wide_ResNet_28_2_tofd(seed=30,num_classes=100)
from general_utils import get_optimizer_scheduler

optimizer,scheduler = get_optimizer_scheduler(net)



train_grid_regular_ce(model=net,optimizer=optimizer,
                      scheduler=scheduler,server=2,experiment_name="Type_2_tofd_wres_teacher_3_out",specific_file_name="_",
                      additional_note="",multiple_gpu=[0],train_on="cuda:0")