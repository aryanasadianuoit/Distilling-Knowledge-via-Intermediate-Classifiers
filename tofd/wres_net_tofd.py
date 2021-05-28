import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from general_utils import reproducible_state

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)

        #self.auxiliary1 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)



        self.auxiliary1 = nn.Sequential(self.layer2,
                                        self.layer3
                                        )

        self.auxiliary2 = nn.Sequential(self.layer3)



        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.linear1 = nn.Linear(nStages[3], num_classes)
        self.linear2 = nn.Linear(nStages[3], num_classes)
        #self.linear3 = nn.Linear(nStages[3], num_classes)



    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)


        layer_1_out = self.layer1(out)
        out1_feature = self.auxiliary1(layer_1_out)
        out1_feature = F.relu(self.bn1(out1_feature))
        out1_feature = nn.AdaptiveAvgPool2d((1,1))(out1_feature)
        out1_feature = out1_feature.view(out1_feature.size(0), -1)

        layer_2_out = self.layer2(layer_1_out)
        out2_feature = self.auxiliary2(layer_2_out)
        out2_feature = F.relu(self.bn1(out2_feature))
        out2_feature = nn.AdaptiveAvgPool2d((1, 1))(out2_feature)
        out2_feature = out2_feature.view(out2_feature.size(0), -1)

        layer_3_out = self.layer3(layer_2_out)
        out3_feature = F.relu(self.bn1(layer_3_out))
        out3_feature = nn.AdaptiveAvgPool2d((1,1))(out3_feature)
        out3_feature = out3_feature.view(out3_feature.size(0), -1)


        out1 = self.linear1(out1_feature)
        out2 = self.linear2(out2_feature)
        out3 = self.linear(out3_feature)

        return [out3, out2, out1],[out3_feature,out2_feature,out1_feature]


def get_Wide_ResNet_28_2_tofd(seed=30,num_classes=100):

    reproducible_state(seed=seed)
    net = Wide_ResNet(28, 2, 0.3, num_classes= num_classes)
    from torchsummary import summary
    summary(net, input_size=(3, 32, 32), device="cpu")
    #summary(net, input_size=(3, 32, 32), device="cpu")
    #summary(net.layer3, input_size=(64, 16, 16), device="cpu")
    #virtualin = torch.rand((1,3,32,32))
    #outs = net(virtualin)
    #for out in outs:
        #print(out.shape)
    return net

#net = get_Wide_ResNet_28_2_tofd()



def get_Wide_ResNet_16_2(seed=30,num_classes=100):

    reproducible_state(seed=seed)
    net = Wide_ResNet(16, 2, 0.3, num_classes= num_classes)
    from torchsummary import summary
    summary(net,input_size=(3,32,32),device="cpu")

    return net

"""
test = get_Wide_ResNet_28_10(num_classes=3)

virtual_input = torch.rand((1,3,32,32))
outs = test(virtual_input)

for output in outs:
    print(output.size())

from models.Middle_Logit_Gen import Model_Wrapper,Middle_Logit_Generator
from torchsummary import summary
from general_utils import total_number_of_params,intermediate_ouput_sizer

summary(test,input_size=(3,32,32),device="cpu")
params_layer1 = total_number_of_params(test.conv1) + total_number_of_params(test.bn1)+ total_number_of_params(test.layer1)
params_layer2 = params_layer1 + total_number_of_params(test.layer2)
params_layer3 = params_layer2 + total_number_of_params(test.layer3)

print("Total # Params  Up to Layer 1 ===> ",params_layer1)
print("Total # Params  Up to Layer 2 ===> ",params_layer2)
print("Total # Params  Up to Layer 3 ===> ",params_layer3)


layer1_model = Middle_Logit_Generator(middle_input=outs[1])
layer2_model = Middle_Logit_Generator(middle_input=outs[2])
layer3_model = Middle_Logit_Generator(middle_input=outs[3])

print("LAYER 1 MODEL ARCITECHTURE ")
summary(layer1_model,input_size=outs[1].size(),device="cpu")
print("\n\n\n","#"*40)

print("LAYER 2 MODEL ARCITECHTURE ")
summary(layer2_model,input_size=outs[2].size(),device="cpu")
print("\n\n\n","#"*40)

print("LAYER 3 MODEL ARCITECHTURE ")
summary(layer3_model,input_size=outs[3].size(),device="cpu")
print("\n\n\n","#"*40)
"""