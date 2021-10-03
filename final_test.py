import torch
import numpy as np
from models_repo.massive_resnets import *
from models_repo.tiny_resnets import *
from models_repo.Middle_Logit_Generator import *
import argparse
from train_funcs import train_regular_ce,\
    train_regular_middle_logits,\
    train_kd_or_fitnets_2,\
    stage_1_fitnet_train,\
    dml_train_regular
from train_dih import train_via_dih
from dih_utils import load_trained_intermediate_heads



parser = argparse.ArgumentParser()
#General training setting
parser.add_argument('--training_type', default='dih', type=str,
                    help='The mode for training, could be either "ce" (regular cross-entropy)'
                         ' "kd" (canonical knowledge distillation) "fine_tune" (fine_tuning the intermediate heads) "fitnets", "dml" (deep mutual learning) or dih. default =  "dih"')
parser.add_argument('--epochs', default=240, type=int, help='Input the number of epochs: default(240)')
parser.add_argument('--momentum', default=0.9, type=float, help='Input the momentum: default(0.9)')
parser.add_argument('--nesterov', default=True)
parser.add_argument('--no-nesterov', action='store_false', dest='nesterov', help='Disable Nesterov: default(True)')
parser.add_argument('--batch_size', default=128, type=int, help='Input the batch size: default(128)')
parser.add_argument('--lr', default=0.05, type=float, help='Input the learning rate: default(0.05)')
parser.add_argument('--wd', default=5e-4, type=float, help='Input the weight decay rate: default(5e-4)')
parser.add_argument('--schedule', nargs='+', type=int, default=[150, 180, 210],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--schedule_gamma', type=float, default=0.2,
                    help='multiply the learning rate to this factor at pre-defined epochs in schedule (default : 0.2)')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='Input the name of dataset: default(CIFAR100)')
parser.add_argument('--student', default='res8', type=str, help='The student model. default: ResNet 8')
parser.add_argument('--teacher', default=None, type=str, help='The teacher model. default: ResNet 110')
parser.add_argument('--path_to_save', default='./model.pth', type=str,
                    help='the path to save the model and/or headers after training')
parser.add_argument('--saved_path', default='/model.pth', type=str,
                    help='the path of the saved model')
parser.add_argument('--saved_intermediates_directory', default='./saved_headers/', type=str,
                    help='the directory of fined-tuned mounted intermediate heads')

parser.add_argument('--gpu_id', default='cuda:0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--kd_alpha', default=0.1, type=float, help='alpha weigth in knowedge distiilation loss function')
parser.add_argument('--kd_temperature', default=5, type=int, help='Temperature in knowedge distiilation loss function')
parser.add_argument('--seed', default=3, type=int, help='seed value for reproducibility')


#FitNets stage 1

parser.add_argument('--student_stage_1_saved', default='/model.pth', type=str,
                    help='the path of the saved partial student upto the guided layer (stage 1 of FitNets)')
parser.add_argument('--epochs_fitnets_1', default=40, type=int, help='Input the number of epochs: default(40) FitNets stage 1')
parser.add_argument('--momentum_fitnets_1', default=0.9, type=float, help='Input the momentum: default(0.9) FitNets stage 1')
parser.add_argument('--nesterov_fitnets_1', default=True, type=bool, help='Input the status of nesterov: default(True) FitNets stage 1')
parser.add_argument('--lr_fitnets_1', default=0.1, type=float, help='Input the learning rate: default(0.1) FitNets stage 1')
parser.add_argument('--wd_fitnets_1', default=5e-4, type=float, help='Input the weight decay rate: default(5e-4) FitNets stage 1')
parser.add_argument('--schedule_fitnets_1', type=int, nargs='+', default=[60, 120, 180],
                    help='Decrease learning rate at these epochs. FitNets stage 1')
parser.add_argument('--schedule_gamma_fitnets_1', type=float, default=0.2,
                    help='multiply the learning rate to this factor at pre-defined epochs in schedule (default : 0.2) FitNets stage 1')


# Create a dictionary (key,value) pair with the arguments
# state = {'batch_size': 64, 'dataset': 'cifar100', 'epochs': 200, 'epochs_fitnets_1': 40, 'gpu_id': 'cuda:0', 'kd_alpha': 0.1, 'kd_temperature': 5, 'lr': 0.1, 'lr_fitnets_1': 0.1, 'momentum': 0.9, 'momentum_fitnets_1': 0.9, 'nesterov': True, 'nesterov_fitnets_1': True, 'path_to_save': './teacher_res8_cifar100_seed_3_epochs_200.th', 'saved_intermediates_directory': './saved_headers/', 'saved_path': '/model.pth', 'schedule': [60, 120, 180], 'schedule_fitnets_1': [60, 120, 180], 'schedule_gamma': 0.2, 'schedule_gamma_fitnets_1': 0.2, 'seed': 3, 'student': 'res8', 'student_stage_1_saved': '/model.pth', 'teacher': 'res8', 'training_type': 'ce', 'wd': 0.0005, 'wd_fitnets_1': 0.0005}
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Print the arguments
for (arg,value) in state.items():
    print(arg+" : "+str(value)+"\n"+"*"*30)

# Models for this experiment
models_dict = {"res8": resnet8_cifar,
               "res14": resnet14_cifar,
               "res20": resnet20_cifar,
               "res110": resnet110_cifar,
               "res34": ResNet34,
               "res18": ResNet18}


# The number of mounted intermediate heads based on the model architecture in this paper.
intermediate_heads_quantity = {"res8": 3,
               "res14": 3,
               "res20": 3,
               "res110": 3,
               "res34": 4,
               "res18": 4}

# Output classes
if args.dataset == "cifar10":
    num_classes = 10
else:
    num_classes = 100

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Regular Cross-entrophy (no distillation)
# args.teacher = teacher architecture (res8, res110...)
if args.teacher != None:
    # teacher = resnet8_cifar (PyTorch nn.module)
    teacher = models_dict[args.teacher](num_classes=num_classes)

    # regular cross_entropy for the teacher
    if args.training_type == "ce":
# state = {'batch_size': 64, 'dataset': 'cifar100', 'epochs': 200, 'epochs_fitnets_1': 40, 'gpu_id': 'cuda:0', 'kd_alpha': 0.1, 'kd_temperature': 5, 'lr': 0.1, 'lr_fitnets_1': 0.1, 'momentum': 0.9, 'momentum_fitnets_1': 0.9, 'nesterov': True, 'nesterov_fitnets_1': True, 'path_to_save': './teacher_res8_cifar100_seed_3_epochs_200.th', 'saved_intermediates_directory': './saved_headers/', 'saved_path': '/model.pth', 'schedule': [60, 120, 180], 'schedule_fitnets_1': [60, 120, 180], 'schedule_gamma': 0.2, 'schedule_gamma_fitnets_1': 0.2, 'seed': 3, 'student': 'res8', 'student_stage_1_saved': '/model.pth', 'teacher': 'res8', 'training_type': 'ce', 'wd': 0.0005, 'wd_fitnets_1': 0.0005}
        optimizer = torch.optim.SGD(teacher.parameters(),
                                    lr=args.lr,             # 'lr': 0.1
                                    weight_decay=args.wd,   # 'wd': 0.0005
                                    momentum=args.momentum, # 'momentum': 0.9
                                    nesterov=args.nesterov) # 'nesterov': True

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[60, 120, 180], # args.schedule,  # 'schedule': [60, 120, 180]
                                                         gamma=args.schedule_gamma, # 'schedule_gamma': 0.2
                                                         last_epoch=-1)

        # return is used for nothing
        # train_funcs::train_regular_ce
        trained_model = train_regular_ce(model=teacher,
                                         optimizer=optimizer,
                                         epochs=args.epochs,
                                         dataset=args.dataset,
                                         train_on=args.gpu_id,
                                         batch_size=args.batch_size,
                                         scheduler=scheduler,
                                         seed=args.seed,
                                         path_to_save=args.path_to_save)

    elif args.training_type == "fine_tune":  #Fine_Tuning the mounted intermedeiate headers

        saved_state_dict = torch.load(args.saved_path)
        teacher.to(args.gpu_id)
        testing_state_dict = {}
        for (key, value), (key_saved, value_saved) in zip(teacher.state_dict().items(), saved_state_dict.items()):
            testing_state_dict[key] = value_saved
        teacher.load_state_dict(testing_state_dict)
        teacher.eval()
        #using a random virtual input in size of our dataset's input images(3,32,32) in order to exploit intermediate outputs of the teacher.
        virtual_input = torch.rand((1, 3, 32, 32),device=args.gpu_id)
        outputs = teacher(virtual_input)
        intermediate_classifier_models = {}

        for mounted_head_index in range(intermediate_heads_quantity[args.teacher]):
            # create intermediate classifier modules which can be mounted in different depth of the core teacher model.
            intermediate_classifier_models[mounted_head_index+1] = Middle_Logit_Generator(outputs[mounted_head_index+1], num_classes=num_classes)

        total_internal_heads_params = []  #sum all the trainable parameters in all of the mounted intermediate heads
        for classifier in intermediate_classifier_models.values():
            total_internal_heads_params += (list(classifier.parameters()))


        optimizer_combined = torch.optim.SGD(total_internal_heads_params,
                                             lr=args.lr,
                                             weight_decay=args.wd,
                                             momentum=args.momentum,
                                             nesterov=args.nesterov)
        scheduler_combined = torch.optim.lr_scheduler.MultiStepLR(optimizer_combined,
                                                                  milestones=args.schedule,
                                                                  gamma=args.schedule_gamma,
                                                                  last_epoch=-1)
        #fine_tuning the added intermediate headers
        train_regular_middle_logits(model=teacher,
                                    optimizer=optimizer_combined,
                                    path_to_save=args.path_to_save,
                                    middle_logits_model_dict=intermediate_classifier_models,
                                    epochs=args.epochs,
                                    train_on=args.gpu_id,
                                    scheduler=scheduler_combined,
                                    batch_size=args.batch_size,
                                    dataset=args.dataset)




    elif args.training_type == "dih": #DIH Distillation
        if args.student != None:
            student = models_dict[args.student](num_classes=num_classes)
            optimizer = torch.optim.SGD(student.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.wd,
                                        momentum = args.momentum,
                                        nesterov = args.nesterov)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=args.schedule,
                                                             gamma=args.schedule_gamma,
                                                             last_epoch=-1)

            #load the fine_tuned intermediate headers
            trained_intermediate_heads = load_trained_intermediate_heads(core_model=teacher,
                                                                         core_model_saved_path=args.saved_path,
                                                                         heads_directory=args.saved_intermediates_directory,
                                                                         num_classes=num_classes)
            #DIH distillation
            train_via_dih(student=student,
                          trained_core_teacher=teacher,
                          traind_intermediate_classifers_dict=trained_intermediate_heads,
                          optimizer=optimizer,
                          dataset=args.dataset,
                          path_to_save=args.path_to_save,
                          epochs=args.epochs,
                          device_to_train_on= args.gpu_id,
                          input_sample_size=(args.batch_size, 32, 32),
                          multiple_gpu=None,
                          scheduler=scheduler,
                          kd_alpha=args.kd_alpha,
                          kd_temperature=args.kd_temperature,
                          seed=args.seed)

    elif args.training_type == "kd":
        if args.student != None:

            student = models_dict[args.student](num_classes=num_classes)

            if args.saved_path != None:
                temp_dict = {}
                teacher_saved_state_dict = torch.load(args.saved_path)
                for (key,_),(key_saved,value_saved) in zip(teacher.state_dict().items(),teacher_saved_state_dict.items()):
                    if "module."+ key == key_saved:
                        temp_dict[key] = value_saved
                        teacher.load_state_dict(temp_dict)
                        teacher.eval()

            optimizer = torch.optim.SGD(student.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.wd,
                                        momentum = args.momentum,
                                        nesterov = args.nesterov)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=args.schedule,
                                                             gamma=args.schedule_gamma,
                                                             last_epoch=-1)

            train_kd_or_fitnets_2(student=student,
                                trained_teacher=teacher,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                input_data_size=(args.batch_size, 32, 32),
                                kd_alpha=args.kd_alpha,
                                kd_temperature=args.kd_temperature,
                                seed=args.seed,
                                epochs=args.epochs,
                                train_on=args.gpu_id,
                                dataset=args.dataset,
                                path_to_save=args.path_to_save)



    elif args.training_type == "fitnets":
        if args.student != None:

            student = models_dict[args.student](num_classes=num_classes)

            if args.saved_path != None:
                temp_dict = {}
                teacher_saved_state_dict = torch.load(args.saved_path)
                for (key, _), (key_saved, value_saved) in zip(teacher.state_dict().items(),
                                                              teacher_saved_state_dict.items()):
                    if "module." + key == key_saved:
                        temp_dict[key] = value_saved
                        teacher.load_state_dict(temp_dict)
                        teacher.eval()

            optimizer = torch.optim.SGD(student.parameters(),
                                        lr=args.lr_fitnets_1,
                                        weight_decay=args.wd_fitnets_1,
                                        momentum=args.momentum_fitnets_1,
                                        nesterov=args.nesterov_fitnets_1)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=args.schedule_fitnets_1,
                                                             gamma=args.schedule_gamma_fitnets_1,
                                                             last_epoch=-1)

            teacher_path = args.saved_path
            saved_state_dict = torch.load(teacher_path)
            testing_state_dict = {}
            for (key, value), (key_saved, value_saved) in zip(teacher.state_dict().items(),
                                                              saved_state_dict.items()):
                testing_state_dict[key] = value_saved
            teacher.load_state_dict(testing_state_dict)
            teacher.eval()

            frozen_student_modules = [student.avgpool, student.fc, student.layer3]


            partial_student_satet_dict = stage_1_fitnet_train(partial_student=student,
                                                              frozen_student_modules=frozen_student_modules,
                                                              partail_teacher=teacher,
                                                              guided_layer=None,
                                                              optimizer=optimizer,
                                                              path_to_save=args.path_to_save,
                                                              dataset=args.dataset,
                                                              epochs=args.epochs_fitnets_1,
                                                              train_on=args.gpu_id,
                                                              scheduler=scheduler,
                                                              input_data_size=(args.batch_size, 32, 32))

            student_temp_weight = {}
            for (key, value), (key_saved, value_saved) in zip(student.state_dict().items(),
                                                              partial_student_satet_dict.items()):
                student_temp_weight[key] = value_saved
            student.load_state_dict(student_temp_weight)

            optimizer = torch.optim.SGD(student.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.wd,
                                        momentum=args.momentum,
                                        nesterov=args.nesterov)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=args.schedule,
                                                             gamma=args.schedule_gamma,
                                                             last_epoch=-1)

            train_kd_or_fitnets_2(student=student,
                                  trained_teacher=teacher,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  input_data_size=(args.batch_size, 32, 32),
                                  kd_alpha=args.kd_alpha,
                                  kd_temperature=args.kd_temperature,
                                  seed=args.seed,
                                  epochs=args.epochs,
                                  train_on=args.gpu_id,
                                  dataset=args.dataset,
                                  path_to_save=args.path_to_save)

    elif args.training_type == "dml":

        if args.student != None:
            student = models_dict[args.student](num_classes=num_classes)
            peer1 = student

        peer2 = teacher

        optimizer_peer1 = torch.optim.SGD(student.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.wd,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov)
        peer1_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_peer1,
                                                         milestones=args.schedule,
                                                         gamma=args.schedule_gamma,
                                                         last_epoch=-1)

        optimizer_peer2 = torch.optim.SGD(teacher.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.wd,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov)
        peer2_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_peer2,
                                                         milestones=args.schedule,
                                                         gamma=args.schedule_gamma,
                                                         last_epoch=-1)

        peers = {}
        peers["peer1"] = peer1
        peers["peer2"] = peer2

        optimizers = {}
        optimizers["peer1"] = optimizer_peer1
        optimizers["peer2"] = optimizer_peer2

        schedulers = {}
        schedulers["peer1"] = peer1_scheduler
        schedulers["peer2"] = peer2_scheduler

        kd_temperature_dict = {}
        kd_temperature_dict["peer1"] = 1.0
        kd_temperature_dict["peer2"] = 1.0
        kd_alpha_dict = {}
        kd_alpha_dict["peer1"] = 0.1
        kd_alpha_dict["peer2"] = 0.1

        dml_train_regular(peers=peers,
                  optimizers=optimizers,
                  train_on=args.gpu_id,
                  dataset=args.dataset,
                  scheduler=schedulers,
                  alpha_dict=kd_alpha_dict,
                  temperature_dict=kd_temperature_dict,
                  path_directory_to_save=args.path_to_save,data_input_size=(args.batch_size, 32, 32),
                  seed = args.seed)






