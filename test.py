import torch
from models_repo.massive_resnets import *
from models_repo.tiny_resnets import *
from models_repo.Middle_Logit_Generator import *
import argparse
from train_funcs import train_regular_ce, train_regular_middle_logits
from train_dih import train_via_dih
from dih_utils import load_trained_intermediate_heads



parser = argparse.ArgumentParser()
#General training setting
parser.add_argument('--training_type', default='dih', type=str,
                    help='The mode for training, could be either "ce" (regular cross-entropy)'
                         ' "fine_tune" (fine_tuning the intermediate heads) or dih. default =  "dih"')
parser.add_argument('--epochs', default=200, type=int, help='Input the number of epochs: default(200)')
parser.add_argument('--momentum', default=0.9, type=float, help='Input the momentum: default(0.9)')
parser.add_argument('--nesterov', default=True, type=float, help='Input the status of nesterov: default(True)')
parser.add_argument('--batch_size', default=64, type=int, help='Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help='Input the learning rate: default(0.1)')
parser.add_argument('--wd', default=5e-4, type=float, help='Input the weight decay rate: default(5e-4)')
parser.add_argument('--schedule', type=list, nargs='+', default=[60, 120, 180],
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


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

for (arg,value) in state.items():
    print(arg+" : "+str(value)+"\n"+"*"*30)
#print(args)


#Models for this experiment
models_dict = {"res8": resnet8_cifar,
               "res14": resnet14_cifar,
               "res20": resnet20_cifar,
               "res110": resnet110_cifar,
               "res34": ResNet34,
               "res18": ResNet18}


#The number of mounted intermediate heads based on the model architecture in this paper.
intermediate_heads_quantity = {"res8": 3,
               "res14": 3,
               "res20": 3,
               "res110": 3,
               "res34": 4,
               "res18": 4}

if args.dataset == "cifar10":
    num_classes = 10
else:
    num_classes = 100

if args.teacher != None:
    teacher = models_dict[args.teacher](num_classes=num_classes)
    if args.training_type == "ce":  #regular cross_entropy for the teacher
        optimizer = torch.optim.SGD(teacher.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.wd,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=args.schedule,
                                                         gamma=args.schedule_gamma,
                                                         last_epoch=-1)

        trained_model = train_regular_ce(model=teacher,
                                         optimizer=optimizer,
                                         epochs=args.epochs,
                                         scheduler=scheduler,
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






