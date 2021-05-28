import torch.nn as nn
from general_utils import reproducible_state
import torch.nn.functional as  F

class Middle_Logit_Generator(nn.Module):

    def __init__(self,middle_input,seed=3 ,num_classes = 100):
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
        x = F.adaptive_avg_pool2d(x, (1, 1)
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


