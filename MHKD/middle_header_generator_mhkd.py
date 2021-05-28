import torch.nn as nn
from general_utils import reproducible_state
import torch.nn.functional as  F

class Middle_Logit_Generator(nn.Module):

    def __init__(self,middle_input,seed=3 ,num_classes = 10):
        reproducible_state(seed=seed)
        super(Middle_Logit_Generator, self).__init__()
        self.added_linear = nn.Linear(middle_input.size(1) * middle_input.size(2) * middle_input.size(3), num_classes)

    def forward(self, middle_input):
        fllaten_added = middle_input.view(middle_input.size(0), -1)
        added_linear_output = self.added_linear(fllaten_added)
        return added_linear_output





class Middle_Logit_Generator_mhkd(nn.Module):

    def __init__(self,middle_input,seed=3 ,num_classes = 10,middle_dimension=256,padding = 0):
        reproducible_state(seed=seed)
        super(Middle_Logit_Generator_mhkd, self).__init__()
        self.conv_1 = nn.Conv2d(middle_input.size(1) , middle_dimension, kernel_size=3, bias=False,padding=padding)
        self.bn_1 = nn.BatchNorm2d(middle_dimension)
        self.conv_2 = nn.Conv2d(middle_dimension , middle_dimension, kernel_size=3, bias=False)
        self.bn_2 = nn.BatchNorm2d(middle_dimension)
        #self.avg_pool =F.adaptive_avg_pool2d(x, (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.linear_1 = nn.Linear( middle_dimension, middle_dimension)
        self.linear_2 = nn.Linear( middle_dimension, num_classes)

    def forward(self, middle_input):

        x = self.conv_1(middle_input)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        #print("Before AVG Pooling  ===> ",x.shape)
        #x = self.avg_pool(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        #print("After AVG Pooling  ===> ",x.shape)


        x = x.view(x.size(0), -1)

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x





class Guided_Conv1_Generator(nn.Module):

    def __init__(self,middle_input,seed=3 ,num_classes = 10):
        reproducible_state(seed=seed)
        super(Guided_Conv1_Generator, self).__init__()
        self.guided = nn.Linear(middle_input.size(1) * middle_input.size(2) * middle_input.size(3), num_classes)

    def forward(self, middle_input):
        added_linear_output = self.added_linear(middle_input)
        return added_linear_output



class Model_Wrapper(nn.Module):

    def __init__(self, core_model,attached_part,core_output_index =1):
        super(Model_Wrapper, self).__init__()
        self.core = core_model
        self.attached_part = attached_part
        self.core_output_index = core_output_index

    def forward(self, x):
        out = self.core(x)
        if not isinstance(self.core_output_index,list):
            out = self.attached_part(out[self.core_output_index])
        else:
            out = self.attached_part(out[self.core_output_index[0]][self.core_output_index[1]-1])
        return out


"""

from models.VGG_models import VGG_Intermediate_Branches
from torchsummary import summary
import torch
test = VGG_Intermediate_Branches("VGG11",num_classes=100)

virtual_input = torch.rand((1,3,32,32))


outputs = test(virtual_input)



for output in outputs:
    print(output.shape)

print("*"*30,"\n")

added_module_1 = Middle_Logit_Generator_mhkd(middle_input=outputs[1],num_classes=100,middle_dimension=256)

summary(added_module_1,input_size=(128, 8 , 8),device="cpu")

"""