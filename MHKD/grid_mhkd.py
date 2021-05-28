import torch
from mhkd_training import train_mhkd_grid
from models.resnet_cifar import resnet8_cifar, resnet110_cifar,resnet20_cifar
from models.Middle_Logit_Gen import Model_Wrapper
from dataloader import get_test_loader_cifar
from middle_header_generator_mhkd import Middle_Logit_Generator_mhkd
import os
from general_utils import get_optimizer_scheduler

DATASETS = ["cifar100"]
DEVICE = "cuda:1"

log_path = "/home/mhkd/"

seeds = [30,50,67]
mhkd_beta = 0.5

temperatures = [4,]
alphas = [0.1]
SPATIAL_SIZE = 32
BATCH_SIZE = 64
EPOCHS= 200
SERVER = 2

TEST_MODE = True


for dataset in DATASETS:

    if dataset == "cifar10":
        NUM_CLASSES = 10
        test_loader = get_test_loader_cifar(batch_size=BATCH_SIZE, dataset=dataset)

        teacher_path = "/home/teacher/cifar10/res_110_teacher.pth"


    elif dataset == "cifar100":
        NUM_CLASSES = 100
        test_loader = get_test_loader_cifar(batch_size=BATCH_SIZE, dataset=dataset)
        teacher_path = "/home/teacher/cifar100/res_110_teacher.pth"  Res110


    for seed in seeds:

        virtual_input = torch.rand((1, 3, SPATIAL_SIZE, SPATIAL_SIZE))

        student = resnet20_cifar(seed=seed,num_classes=NUM_CLASSES)

        student_outs = student(virtual_input)

        student_head_1_model = Middle_Logit_Generator_mhkd(student_outs[1], num_classes=NUM_CLASSES, seed=seed)
        student_head_2_model = Middle_Logit_Generator_mhkd(student_outs[2], num_classes=NUM_CLASSES, seed=seed)
        #student_head_3_model = Middle_Logit_Generator_mhkd(student_outs[3], num_classes=NUM_CLASSES, seed=seed)


        student_headers_dict = {}
        student_headers_dict[1] = student_head_1_model
        student_headers_dict[2] = student_head_2_model


        teacher_core = VGG_Intermediate_Branches("VGG11",seed=seed,num_classes=NUM_CLASSES)
        full_modules_state_dict = {}
        saved_state_dict = torch.load(teacher_path)
        testing_state_dict = {}
        for (key, value), (key_saved, value_saved) in zip(teacher_core.state_dict().items(), saved_state_dict.items()):
            testing_state_dict[key] = value_saved
            full_modules_state_dict["core."+key] = value_saved
        teacher_core.load_state_dict(testing_state_dict)
        teacher_core.eval()



        teachers_outs = teacher_core(virtual_input)

 
        teacher_head_1_model = Middle_Logit_Generator_mhkd(teachers_outs[0],num_classes=NUM_CLASSES,seed=seed)
        teacher_head_2_model = Middle_Logit_Generator_mhkd(teachers_outs[1],num_classes=NUM_CLASSES,seed=seed)





        teacher_headers_dict = {}
        teacher_headers_dict[1] = teacher_head_1_model
        teacher_headers_dict[2] = teacher_head_2_model


        params = list(student.parameters()) + \
                 list(student_head_1_model.parameters()) + \
                 list(student_head_2_model.parameters()) + \
                 list(teacher_head_1_model.parameters()) + \
                 list(teacher_head_2_model.parameters())
      



        optimizer, scheduler = get_optimizer_scheduler(params, params_sent=True)

        for temperature in temperatures:
            for alpha in alphas:
                kd_alpha = {}
                # FINAL LOGITS
                kd_alpha[0] = alpha
                # WEAKEST CLASSIFER
                kd_alpha[1] = alpha
                kd_alpha[2] = alpha
   

                experiment_name = dataset + "_seed_" + str(seed) + "_temp_" + str(temperature) + "_alpha_" + str(alpha)
                test_acc,time_elapsed = train_mhkd_grid(student=student,
                                          trained_core_teacher=teacher_core,
                                          teacher_headers_dict=teacher_headers_dict,
                                          student_headers_dict=student_headers_dict,
                                          mhkd_beta = mhkd_beta,
                                          optimizer=optimizer,
                                          dataset=dataset,
                                          path_to_save=log_path + experiment_name+".pth",
                                          epochs=EPOCHS,
                                          train_on=DEVICE,
                                          server=SERVER,
                                          input_sample_size=(BATCH_SIZE, SPATIAL_SIZE, SPATIAL_SIZE),
                                          scheduler=scheduler,
                                          kd_alpha=kd_alpha,
                                          kd_temperature=temperature)

                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60),"\t","Dataset ==>", dataset, "\tSeed ==>", seed, "\tTemperature ==>", temperature, "\tAlpha ===>",
                      alpha, "\tTest Acc ==>", test_acc)



                log_text = "Experiment Name : " + experiment_name + "\n"
                if not os.path.exists(log_path + "/res8.txt"):
                    readme = open(log_path + "/res8.txt", "a+")


                else:
                    readme = open(log_path + "/res8.txt", "a+")

                log_text += "Test Acc ==> " + str(test_acc) + "\n"
                log_text += ("#" * 40) + "\n\n"
                readme.write(log_text)
                readme.close()
