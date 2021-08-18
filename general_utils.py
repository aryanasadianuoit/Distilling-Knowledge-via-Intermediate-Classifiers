import math

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import os.path
import  matplotlib.pyplot as plt
import kd_loss
from tqdm import tqdm
import numpy as np
import os
import datetime
import time
from varname import nameof

IMAGE_NET_SIZE = 224
REGRESSOR_TYPE_CONV = 1
REGRESSOR_TYPE_LINEAR = 2
CRITERION_CROSS_ENTROPY =  1
CRITERION_MSE =  2
CRITERION_KLDIVLOSS =  3
OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1
NUM_CLASSES_CIFAR_10 = 10
TEMPERATURE = 4
Epsilon = 0.1

SERVER_1_PREFIX_PATH = "/server1/saved/"
SERVER_2_PREFIX_PATH = "/server2/saved/"
SERVER_2_GRID_PREFIX_PATH = "/grid/"
SERVER_2_PREFIX_PATH_CODISTILLATION = "/saved/codistillation/"
SERVER_1_IS_AVAILABLE = "1"
SERVER_2_IS_AVAILABLE = "2"
tanh_normalizer = nn.Tanh()
relu_normalizer = nn.ReLU(inplace= True)


def get_criterion(criterion):
    if criterion == CRITERION_CROSS_ENTROPY:
        return  nn.CrossEntropyLoss()
    elif criterion == CRITERION_MSE:
        return nn.MSELoss()
    elif criterion == CRITERION_KLDIVLOSS:
        return nn.KLDivLoss
    else:
        return nn.CrossEntropyLoss()

def get_optimizer(optimizer_type,input_model,lr = 0.001, moemntum = None):
    if optimizer_type == OPTIMIZER_SGD:
        if moemntum != None:
            return optim.SGD(input_model.parameters(), lr=lr, momentum= 0.9)
        elif optimizer_type == OPTIMIZER_ADAM:
            return optim.Adam(input_model.parameters(), lr=lr)


def dist_loss(teacher_logits,temp, student_logits):
    prob_t = F.softmax(teacher_logits / temp, dim=1)
    log_prob_s = F.log_softmax(student_logits / temp, dim=1)
    dist_loss = -(prob_t * log_prob_s).sum(dim=1).mean()
    return dist_loss
#    return dist_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def test_data_evaluation(model,
                         test_loader,
                         device = "cuda",
                         saved_load_state_path = None,
                         state_dict = None,
                         show_log=False,
                         server = None,
                         state_dict_loaded = False,
                         branch_model_saved_path = None,
                         branch_model_saved_state_dict = None,):
    """

    :param model: the target model that needs to be tested and be evaluated
    :param saved_load_state_path: Optional(Default:None) if the model has already been trained,
     the saved state dict can be passed
    :param test_loader: test loader
    :param device: either "cuda" for doing the evaluation on GPU , or "cpu" for evaluation on CPU
    :return:
    """
    evaluation_log = ""
    max_test_acc = 0.0
    max_test_acc_batch_index = 1
    if state_dict_loaded == False:
        if saved_load_state_path != None:
            if server !=None:
                if server == 1:
                    saved_path_address = SERVER_1_PREFIX_PATH+"/"+saved_load_state_path
                elif server == 2:
                    saved_path_address = SERVER_2_PREFIX_PATH_CODISTILLATION+"/"+saved_load_state_path
            else:
                saved_path_address = saved_load_state_path

            temp_dict = {}
            saved_state_dict = torch.load(saved_path_address)
            for (model_key, model_value) , (saved_key,saved_value) in zip(model.state_dict().items(),saved_state_dict.items()):
                if model_key == saved_key:
                    temp_dict[model_key] = saved_state_dict[saved_key]

                else:
                    temp_dict[model_key] = saved_state_dict[saved_key]

            model.load_state_dict(temp_dict)

        #model.load_state_dict(temp_dict)
        if state_dict != None:
            model.load_state_dict(state_dict= state_dict)
            print("Loading the best state dict for evaluation!")
    else:
        print("State Dict Loaded True")

        #model.core.load_state_dict(state_dict=state_dict)
        #branch_models_dict = {}

        if branch_model_saved_path != None:

            model.attached_part.load_state_dict(torch.load(branch_model_saved_path))
        elif branch_model_saved_state_dict != None:
            temp_dict ={}
            for (key,value) in model.attached_part.state_dict().items():
                temp_dict[key] = branch_model_saved_state_dict["module."+key]

            model.attached_part.load_state_dict(temp_dict)

    since = time.time()
    model.eval()
    device_generated = torch.device(device)
    if "cuda" in device:
        model.to(device_generated)
        #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    else:
        device = "cpu"

    print("Test Evaluation In Process.... In ==> ",device)
    correct = 0
    total = 0
    with torch.no_grad():
        counter = 1
        for data in  test_loader:
            images, labels = data
            images, labels = images.to(device_generated), labels.to(device_generated)
            outputs = model(images)
            if isinstance(outputs,tuple):
                # TODO change outputs[1] to outputs[0]
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_acc = (100 * (float(correct) / total))
            if show_log:
                print("batch ===>",counter," Test Acc ===> ",test_acc)
            counter += 1

        time_elapsed = time.time() - since
        print('Evaluation Completed!')
        evaluation_log += "Test Acc ==> %.2f "%test_acc +"\t\t"+'Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(evaluation_log)
        return evaluation_log,test_acc

def test_data_evaluation_multi_classifier(main_model,
                                          axu_models_dict,
                                          test_loader,
                                          device = "cuda",
                                          saved_load_state_path = None,
                                          state_dict = None,
                                          added_regressor = None,
                                          show_log=True,
                                          server = None):
    """

    :param model: the target model that needs to be tested and be evaluated
    :param saved_load_state_path: Optional(Default:None) if the model has already been trained,
     the saved state dict can be passed
    :param test_loader: test loader
    :param device: either "cuda" for doing the evaluation on GPU , or "cpu" for evaluation on CPU
    :return:
    """

    evaluation_log = ""
    max_test_acc = 0.0
    max_test_acc_batch_index = 1
    if saved_load_state_path != None:
        if server !=None:
            if server == 1:
                saved_path_address = SERVER_1_PREFIX_PATH+"/"+saved_load_state_path
            elif server == 2:
                saved_path_address = SERVER_2_PREFIX_PATH_CODISTILLATION+"/"+saved_load_state_path
        else:
            saved_path_address = saved_load_state_path

        temp_dict = {}
        saved_state_dict = torch.load(saved_path_address)
        for (model_key, model_value) , (saved_key,saved_value) in zip(main_model.state_dict().items(),saved_state_dict.items()):
            if model_key == saved_key:
                temp_dict[model_key] = saved_state_dict[saved_key]

            else:

                temp_dict[model_key] = saved_state_dict[saved_key]

        main_model.load_state_dict(temp_dict)

        #model.load_state_dict(temp_dict)
    if state_dict != None:
        main_model.load_state_dict(state_dict= state_dict)
        print("Loading the best state dict for evaluation!")
    since = time.time()
    main_model.eval()
    device_generated = torch.device(device)
    if "cuda" in device:
        main_model.to(device_generated)
        if axu_models_dict != None:
            for models in axu_models_dict.values():
                models.to(device_generated)
    else:
        device = "cpu"

    print("Test Evaluation In Process.... In ==> ",device)
    correct = 0
    total = 0
    preds_dict = {}
    corrects_dict = {}
    test_acc_dict = {}
    with torch.no_grad():
        counter = 1
        for data in  test_loader:
            images, labels = data
            images, labels = images.to(device_generated), labels.to(device_generated)
            outputs = main_model(images)
            if axu_models_dict!= None:
                for index in range(1,len(outputs)):
                    _,preds_dict[index] = torch.max(axu_models_dict[index](outputs[index]).data, 1)
            if isinstance(outputs,tuple):
                # TODO change outputs[1] to outputs[0]
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tmp_dict= corrects_dict.copy()

            for (key,value) in preds_dict.items():
                tmp = (value == labels).sum().item()
                #corrects_dict[key] += (value == labels).sum().item()
                print("tmp ",tmp)
               # print("temp_dict[key] ",temp_dict[key])
                corrects_dict[key] += tmp #+ tmp_dict[key]
                test_acc_dict[key] = (100 * (float(corrects_dict[key]) / total))

            test_acc = (100 * (float(correct) / total))

            if show_log:
                print("batch ===>",counter," Test Acc ===> ",test_acc)
            counter += 1

        time_elapsed = time.time() - since
        print('Evaluation Completed!')
        evaluation_log += "Test Acc ==> %.2f "%test_acc +"\t\t"+'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)

        for (key, value) in test_acc_dict.items():
            print("Layer ",key," Model Test ACC: ",value)

        print(evaluation_log)
        return evaluation_log,test_acc


def experiment_result_saver(model,
                            experiment_name,
                            server,
                            specific_file_name,
                            add_path_to_root = None,
                            model_name = None,
                            additional_note=None,
                            read_me_file_enabled = True,
                            train_acc_dict=None,
                            train_loss_dict=None,
                            val_acc_dict=None,
                            val_loss_dict=None,
                            test_result_log = None,
                            just_update_readme = False):

    if server == 1:
        root_address = SERVER_1_PREFIX_PATH

    elif server == 2:
        root_address = SERVER_2_PREFIX_PATH_CODISTILLATION

    elif server == "grid":
        if add_path_to_root != None:
            root_address = SERVER_2_GRID_PREFIX_PATH+ add_path_to_root
        else:
            root_address = SERVER_2_GRID_PREFIX_PATH

    log_text = "\n"
    current_time = datetime.datetime.now()
    if experiment_name in os.listdir(root_address):
        root_address += "/" + experiment_name

    else:
        root_address += "/" + experiment_name
        os.mkdir(root_address)
        log_text += "Experiment Name : "+experiment_name+"\n"

    if read_me_file_enabled:
        #Create readme.txt for the first time
        if not os.path.exists(root_address+"/readme.txt"):

            log_text += "\n" + "Date : " + current_time.strftime("%c") + "\n\n"
            readme = open(root_address+"/readme.txt","a+")
            log_text += ("#"*50)+"\n\n\n\n"
            readme.write(log_text)

        else:
            readme = open(root_address+"/readme.txt","a+")

        if additional_note != None:
            log_text = "\n" + ("*" * 20) + "Additional Note" + ("*" * 20) + "\t\t =====>\t\t\n" + (
                    "*" * 40) + "\n" + additional_note + "\n\n"
            readme.write(log_text)

        if train_acc_dict or train_loss_dict or val_acc_dict or val_loss_dict != None:
            if model_name != None:
                log_text += "\n\n" + ("#" * 20) + "Experiment Info  Model ===>  "+model_name + ("#" * 20) + "\n\n"
            else:
                log_text += "\n\n" + ("#" * 20) + "Experiment Info" + ("#" * 20) + "\n\n"
            readme.write(log_text)


        if train_acc_dict != None:
            if model_name != None:
                log_text = ("\t" * 3) + nameof(train_acc_dict)+" Model ===> "+model_name + ("\t" * 3) + "\n\n"
            else:
                log_text  = ("\t" * 3) + nameof(train_acc_dict) + ("\t" * 3) + "\n\n"
            log_text += "Epoch \t|\tValue \n"
            log_text += ("*" * 50) +"\n\n"
            for (key,value) in train_acc_dict.items():
                log_text += str(key) + "\t|\t" + str(value)+"\t|\n"
                log_text += ("=" * 30) + "\n"
            readme.write(log_text)


        if train_loss_dict != None:

            if model_name != None:
                log_text = ("\t" * 3) + nameof(train_loss_dict)+" Model ===> "+model_name + ("\t" * 3) + "\n\n"
            else:
                log_text  = ("\t" * 3) + nameof(train_loss_dict) + ("\t" * 3) + "\n\n"

            log_text += "Epoch \t|\tValue \n"
            log_text += "*" * 50 + "\n\n"
            for (key, value) in train_loss_dict.items():
                log_text += str(key) + "\t|\t" + str(value) + "\t|\n"
                log_text += ("=" * 30) + "\n"
            readme.write(log_text)


        if val_acc_dict != None:
            if model_name != None:
                log_text = ("\t" * 3) + nameof(val_acc_dict) + " Model ===> " + model_name + ("\t" * 3) + "\n\n"
            else:
                log_text = ("\t" * 3) + nameof(val_acc_dict) + ("\t" * 3) + "\n\n"

            log_text += "Epoch \t|\tValue \n"
            log_text += "*" * 50 + "\n\n"
            for (key, value) in val_acc_dict.items():
                log_text += str(key) + "\t|\t" + str(value) + "\t|\n"
                log_text += ("=" * 30) + "\n"
            readme.write(log_text)


        if val_loss_dict != None:

            if model_name != None:
                log_text = ("\t" * 3) + nameof(val_loss_dict) + " Model ===> " + model_name + ("\t" * 3) + "\n\n"
            else:
                log_text = ("\t" * 3) + nameof(val_loss_dict) + ("\t" * 3) + "\n\n"

            log_text += "Epoch \t|\tValue \n"
            log_text += "*" * 50 + "\n\n"
            for (key, value) in val_loss_dict.items():
                log_text += str(key) + "\t|\t" + str(value) + "\t|\n"
                log_text += ("=" * 30) + "\n"
            readme.write(log_text)


        if test_result_log != None:

            if model_name != None:
                log_text = ("\n\n"+"*"*30)+" Test Results  Model ===> "+model_name+" :\t" +("*"*30+"\n\n")
            else:
                log_text = ("\n\n" + "*" * 30) + " Test Results" + ("*" * 30 + "\n\n")

            log_text += test_result_log[0]
            readme.write(log_text)

        readme.close()

    if just_update_readme == False:
        complete_saved_path =  root_address+"/"+specific_file_name+".pth"
        torch.save(model.state_dict(), complete_saved_path)
        print("Model ",specific_file_name," Saved! ")
        return complete_saved_path
    else:
        print("Readme file updated. No Model saving!")
        return ""


def plotter(plotting_dicts,server, experiment_name):


    for key,value_dict in plotting_dicts.items():
        #key is a list which teh first element is the name of the model and second element is the name of the value , e.g., validation_acc
        lists = list(value_dict.items())
        key_splitted = key.split("*")

        epochs, main_variable = zip(*lists)  # unpack a list of pairs into two tuples
        plt.ylabel(key_splitted[1], fontsize=12, color='black')

        plt.xlabel('Epochs', fontsize=12, color='black')
        plt.title(key_splitted[0]+" "+key_splitted[1]+' Figure')


        plt.plot(epochs, main_variable, '-gD')

        fig1= plt.gcf()

        plt.show()
        if server == 1:
            root_address = SERVER_1_PREFIX_PATH
        else:
            root_address = SERVER_2_PREFIX_PATH_CODISTILLATION

        if (key_splitted[0]+"_plots") in os.listdir(root_address+experiment_name+"/"):
            print("Plot folder for ",key_splitted[0]," exists!, merging new plots to the folder!")
            root_address += "/" + experiment_name + "/" + (key_splitted[0] + "_plots")
            fig1.savefig(root_address+ "/" + key_splitted[1] + "_" + key_splitted[0] + '.png')

        else:
            root_address += "/" + experiment_name + "/" + (key_splitted[0]+"_plots")
            print("Creating plots_folder for ",str(key_splitted[0]))
            os.mkdir(root_address)
            fig1.savefig(root_address+"/"+key_splitted[1]+"_"+key_splitted[0]+'.png')


def intermediate_ouput_sizer(model,model_module,example_input_tensor = torch.rand((1,3,32,32))):
    """

    :param model: The target model that we want to find its sub_modules intermediate's output dimension
    :param model_module: the specific module or part of the traget model,
    that we want to find each of its children output dimension
    :param example_input_tensor: optional( for 1 round of forward propagation through the model.
     default: batch of size 1 , 3 channels , W & H = 224 (i.e. IMAGENET DATA POINT)
    :return: Dictionary with the key values of children indices in the model_module and values of size of intermediate
    output of each of those indices
    """
    model.eval()
    intermediate_output_dict = {}
    def assign_hooker(target_module,key_value):
        def forward_hooker_internal(self, input, output):
            # input is a tuple of packed inputs
            # output is a Tensor. output.data is the Tensor we are interested
            intermediate_output_dict[key_value] = output.size()
        target_module.register_forward_hook(forward_hooker_internal)




    #for (index, child) in enumerate(model_module.children(),start=0):
    index= 0
    for (_, child) in enumerate(model_module.children(),start=0):
        #print("Type : ",type(child))
        if isinstance(child,torch.nn.Sequential):
            for i in range(0,len(child)):
                #print(" YES index : ",index,"\t\t",type(child),"\t CHild OF CCHILD ==> ",type(child[i]))
                assign_hooker(child[i],index)
                print("Sequential Index ===> ",index)
                index += 1
        else:
            if isinstance(model_module,nn.Sequential):
                assign_hooker(child, index)
            #print("NO index : ", index, "\t\t", type(child))
                index += 1

    _ = model(example_input_tensor)
    return intermediate_output_dict




def get_intermediate_ouput_dict(model,model_module,example_input_tensor = torch.rand((1,3,32,32))):
    """

    :param model: The target model that we want to find its sub_modules intermediate's output dimension
    :param model_module: the specific module or part of the traget model,
    that we want to find each of its children output dimension
    :param example_input_tensor: optional( for 1 round of forward propagation through the model.
     default: batch of size 1 , 3 channels , W & H = 224 (i.e. IMAGENET DATA POINT)
    :return: Dictionary with the key values of children indices in the model_module and values of size of intermediate
    output of each of those indices
    """
    model.eval()
    intermediate_output_dict = {}
    def assign_hooker(target_module,key_value):
        def forward_hooker_internal(self, input, output):
            # input is a tuple of packed inputs
            # output is a Tensor. output.data is the Tensor we are interested
            intermediate_output_dict[key_value] = output
        target_module.register_forward_hook(forward_hooker_internal)




    index= 0
    for (_, child) in enumerate(model_module.children(),start=0):
        #print("Type : ",type(child))
        if isinstance(child,torch.nn.Sequential):
            for i in range(0,len(child)):
                #print(" YES index : ",index,"\t\t",type(child),"\t CHild OF CCHILD ==> ",type(child[i]))
                assign_hooker(child[i],index)
                index += 1
        else:
            assign_hooker(child, index)
            #print("NO index : ", index, "\t\t", type(child))
            index += 1

    _ = model(example_input_tensor)
    return intermediate_output_dict



def additional_note_setter(model_name, weight_initialization, epochs, weight_decay, lr, additional_note,optimizer,criterion):

    additional_note =   "Model Name ====> "+model_name\
                      + "\n" \
                      + "Weight Initialization ===> "+weight_initialization \
    +"\nOptimizer ===> "+ optimizer+"\nCriterion ===> "+criterion+"\nWeight Decay ===> "+weight_decay+"\n# Epochs ===>"+epochs\
    +"\nLearning rate ===> "+lr+"\n Extra points ===>"+additional_note
    return additional_note


def checkpoint_saver(server,experiment_name,model_name,epoch, model_state_dict, optimizer_state_dict,loss,add_path_to_root= None,is_best = False):
    if server == 1:
        root_address = SERVER_1_PREFIX_PATH

    elif server == 2:
        root_address = SERVER_2_PREFIX_PATH_CODISTILLATION
    elif server == "grid":
        if add_path_to_root != None:
            root_address = SERVER_2_GRID_PREFIX_PATH + add_path_to_root
        else:
            root_address = SERVER_2_GRID_PREFIX_PATH
        #root_address = SERVER_2_GRID_PREFIX_PATH

    if experiment_name in os.listdir(root_address):
        #print("Experiment Is Not NEW!  OverWriting!")
        root_address += "/" + experiment_name

    else:
        root_address += "/" + experiment_name
        os.mkdir(root_address)
    if is_best == True:
        checkpoint_path = root_address + "/" + model_name + "_checkpoint_best.pth.tar"
        print("Best model Checkpoint Saved! for model ==> ",model_name )
    else:
        checkpoint_path = root_address+"/"+model_name+"_checkpoint_epoch_"+str(epoch)+".pth.tar"
    #   print("Periodic Checkpoint Saved Epoch ===>",epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss,
    }, checkpoint_path)

    return checkpoint_path


def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d) :
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.xavier_uniform_(m.bias.data)
            print("Not None!")



        #m.bias.data.fill_(0.01)


def he_initialization(m):
    if isinstance(m,nn.Conv2d) :
        torch.nn.init.kaiming_uniform_(m.weight.data,nonlinearity="relu")



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((0,2, 3, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp.squeeze())
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def total_number_of_params(model_or_module):
    """
    :param model_or_module:
    :return:
    """
    total_param_elemenets = 0
    for p in model_or_module.parameters():
        total_param_elemenets += p.numel()

    return total_param_elemenets


def activation_attension(x):
    return F.normalize(x.pow(2).mean(dim=1).view(x.size(0), -1))


def reproducible_state(seed =3,device ="cuda"):
   #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = device
    # fix the seed and make cudnn deterministic
    #seed = 3
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("REPRODUCIBILITY is Active!Random seed = ",seed,"\n")



def grid_logger(experiment_name,
                server,
                gamma_dict,
                temperature_dict,
                test_acc_dict):

    if server == 1:
        root_address = SERVER_1_PREFIX_PATH

    else:
        root_address = SERVER_2_PREFIX_PATH_CODISTILLATION

    log_text = "\n"
    current_time = datetime.datetime.now()
    if experiment_name in os.listdir(root_address):
        root_address += "/" + experiment_name

    else:
        root_address += "/" + experiment_name
        os.mkdir(root_address)
        log_text += "Experiment Name : "+experiment_name+"\n"

    if not os.path.exists(root_address + "/grid_log.txt"):

        log_text += "\n" + "Date : " + current_time.strftime("%c") + "\n\n"
        grid_log = open(root_address + "/grid_log.txt", "a+")
        log_text += ("#" * 50) + "\n\n\n\n"
        grid_log.write(log_text)

    else:
        grid_log = open(root_address + "/grid_log.txt", "a+")

    log_text+="\n"+"*"*15+"<----Gamma---->"+"*"*15+"\n"

    for (key, value) in gamma_dict.items():
        log_text += str(key) + "\t|\t" + str(value) + "\t|\n"
        log_text += ("=" * 30) + "\n"
    grid_log.write(log_text)

    log_text += "\n" + "*" * 15 + "<----Temperature---->" + "*" * 15 + "\n"

    for (key, value) in temperature_dict.items():
        log_text += str(key) + "\t|\t" + str(value) + "\t|\n"
        log_text += ("=" * 30) + "\n"
    grid_log.write(log_text)

    log_text += "\n" + "*" * 15 + "<----Test Accuracy---->" + "*" * 15 + "\n"

    for (key, value) in test_acc_dict.items():
        log_text += str(key) + "\t|\t" + str(value) + "\t|\n"
        log_text += ("=" * 30) + "\n"
    grid_log.write(log_text)

    log_text += "\n\n" +"*" * 30 + "\n\n"

    grid_log.close()



def state_dict_loader(model,saved_path,load_on="cuda",multiple_gpu=[0,1,2,3]):
    saved_state_dict = torch.load(saved_path)

    model.to(load_on)
    if "cuda" in load_on:
        model = torch.nn.DataParallel(model, device_ids=multiple_gpu)
    testing_state_dict = {}
    for key, value in model.state_dict().items():
        # testing_state_dict[key] = saved_state_dict["module." + key]
        testing_state_dict[key] = saved_state_dict[key]

    model.load_state_dict(testing_state_dict)
    model.eval()
    return model




def get_optimizer_scheduler(model_or_params,params_sent = False):
    if params_sent:
        training_params = model_or_params
    else:
        training_params = model_or_params.parameters()

    optimizer = torch.optim.SGD(training_params, lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.2,
                                                     last_epoch=-1)
    return optimizer,scheduler

