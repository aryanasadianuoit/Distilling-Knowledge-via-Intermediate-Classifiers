import os
import torch
from models_repo.Middle_Logit_Generator import Middle_Logit_Generator

def load_trained_intermediate_heads(core_model,core_model_saved_path, heads_directory,num_classes,gpu="cuda:0"):
    saved_state_dict = torch.load(core_model_saved_path)
    core_model.to(gpu)

    testing_state_dict = {}
    for (key, value), (key_saved, value_saved) in zip(core_model.state_dict().items(), saved_state_dict.items()):
        testing_state_dict[key] = value_saved
    core_model.load_state_dict(testing_state_dict) # loading the core model( the model that the heads should be mounted to)
    core_model.eval()  # freeze the core model

    virtual_input = torch.rand((1, 3, 32, 32), device="cuda:0") # a mock input( with the size of real input) for defining the intermediate heads
    outputs = core_model(virtual_input)
    head_paths_list = os.listdir(heads_directory)  # list all the saved intermediate heads models in the entered directory
    new_list = []
    for index in range(len(head_paths_list)):
        if str.endswith(head_paths_list[index], ".pth"):   # just keep the saved models with extension .pth.
            new_list.append(head_paths_list[index])

    list.sort(new_list) # since the intermediate heads are named in order based on the core model output's index. see line 28 of this file

    intermediate_heads = {}
    for index in range (1,len(new_list)+1):
        tmp_header = Middle_Logit_Generator(outputs[index], num_classes=num_classes)

        temp_dict = {}
        for (key, value) in tmp_header.state_dict().items():
            temp_dict[key] = torch.load(heads_directory+new_list[index-1])[key]

        tmp_header.load_state_dict(temp_dict)
        tmp_header.eval()
        intermediate_heads[index] = tmp_header

    print("DONE")
    return intermediate_heads  #return the loaded intermediate heads
