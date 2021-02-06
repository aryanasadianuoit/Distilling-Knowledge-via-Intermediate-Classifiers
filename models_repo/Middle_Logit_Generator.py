import torch.nn as nn

class Middle_Logit_Generator(nn.Module):

    def __init__(self,middle_input,num_classes = 10):
        super(Middle_Logit_Generator, self).__init__()
        self.added_linear = nn.Linear(middle_input.size(1) * middle_input.size(2) * middle_input.size(3), num_classes)

    def forward(self, middle_input):
        fllaten_added = middle_input.view(middle_input.size(0), -1)
        added_linear_output = self.added_linear(fllaten_added)
        return added_linear_output




class Guided_Conv1_Generator(nn.Module):

    def __init__(self,middle_input,num_classes = 10):
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

