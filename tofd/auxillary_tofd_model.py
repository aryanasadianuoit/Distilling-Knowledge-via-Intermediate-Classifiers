import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torchsummary import summary
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from general_utils import reproducible_state

import sys
import numpy as np


#from general_utils import reproducible_state
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


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)



class Linear_Block(nn.Module):
    def __init__(self, num_classes=100,
                 expansion=1, coeeficient=64):
        super(Linear_Block, self).__init__()
        #if num_classes == 200:
            # for Tiny Image_Net

            #self.avgpool = nn.AvgPool2d(16, stride=1)
        #else:
            # for Cifar10-100
            #self.avgpool = nn.AvgPool2d(8, stride=1)

        #self.fc = nn.Sequential(
           # self.avgpool
        self.fc= nn.Linear(coeeficient * expansion, num_classes)
        #)

    def forward(self,x):
        #x = self.avgpool(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        #print("HERE ",x.shape)
        x= x.view(x.size(0), -1)
        x = self.fc(x)
        return x






class AUX_Model_TOFD(nn.Module):

    def __init__(self,config_list,seed=30,kernel_size=3, stride=2, padding=1, affine=True, num_classes=100,
                 expansion=1, coeeficient=64):
        super(AUX_Model_TOFD, self).__init__()
        reproducible_state(seed=seed)


        self.num_classes = num_classes


        self.config_list = config_list

        #if self.config_list != None:
        self.conv_blocks = self._make_layers()
        #else:
            #self.conv_blocks =
        self.fc = Linear_Block(num_classes=self.num_classes,expansion=expansion, coeeficient=coeeficient)

    def forward(self,x):
        x = self.conv_blocks(x)
        #print("x.shape",x.shape)
        x = self.fc(x)
        return x

    def _make_layers(self):

        layers = []

        for  channel_in, channel_out in self.config_list:
            layers.append(SepConv(channel_in, channel_out))




        #for number_of_sep_blocks, channel_in, channel_out in self.config_list:

            #if number_of_sep_blocks == 0:
                #layers.append(Linear_Block(num_classes=self.num_classes))
            #else:
                #for number_of_pairs in range(number_of_sep_blocks):

                    #layers.append(SepConv(channel_in,channel_out))



        return nn.Sequential(*layers)




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

class AUX_Model_TOFD_Wide(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(AUX_Model_TOFD_Wide, self).__init__()
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

        self.auxiliary2 = nn.Sequential(self.layer3
                                       )



        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.linear1 = nn.Linear(nStages[3], num_classes)
        self.linear2 = nn.Linear(nStages[3], num_classes)
        self.linear3 = nn.Linear(nStages[3], num_classes)



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

        layer_2_out = self.layer2(layer_1_out)
        out2_feature = self.auxiliary2(layer_2_out)
        out2_feature = F.relu(self.bn1(out2_feature))
        out2_feature = nn.AdaptiveAvgPool2d((1, 1))(out2_feature)

        layer_3_out = self.layer3(layer_2_out)
        out3_feature = F.relu(self.bn1(layer_3_out))
        out3_feature = nn.AdaptiveAvgPool2d((1, 1))(out3_feature)

        #out = F.avg_pool2d(out, 8)
                                                  #Changed for Tiny ImageNet
        out = F.relu(self.bn1(layer_3_out))
        out = nn.AdaptiveAvgPool2d((1,1))(out)
        out = out.view(out.size(0), -1)
        out4 = self.linear(out)

        out1 = self.linear1(out1_feature.view(out1_feature.size(0), -1))
        out2 = self.linear2(out2_feature.view(out2_feature.size(0), -1))
        out3 =self.linear3(out3_feature.view(out3_feature.size(0), -1))

        return [out4, out3, out2, out1],[out3_feature,out2_feature,out1_feature]


"""
from models.wide_resnet import get_Wide_ResNet_28_2

test = get_Wide_ResNet_28_2(num_classes=100)

virtual_input = torch.rand((1,3,32,32))

outputs = test(virtual_input)

for out in outputs:
    print("Shape",out.shape)


config_list=[(32,64),(64,128),(128,256)]
config_list1=[(32,64)]


model_1 = AUX_Model_TOFD(config_list=config_list,expansion=4)
print("MODEL 1 \n\n")
summary(model_1,input_size=(32,32,32),device="cpu")
model_2 = AUX_Model_TOFD(config_list=config_list[1:],expansion=4)
summary(model_2,input_size=(64,16,16),device="cpu")
model_3 = AUX_Model_TOFD(config_list= config_list[2:],expansion=4)
summary(model_3,input_size=(128,8,8),device="cpu",)
"""
#from models.resnet_cifar import resnet8_cifar

#test = resnet8_cifar(num_classes=100)
#summary(test,input_size=(3,32,32),device="cpu")

#virtual_input = torch.rand((1,3,32,32))

#outputs = test(virtual_input)

#for out in outputs:
  #  print("Shape",out.shape)
#(number of sep blocks, channel_in, channel_out)
#config_list=[(1,16,32),(1,32,64)]
                 #(1,32,64),
                 #(0,0,0),]


#model_1 = AUX_Model_TOFD(config_list=config_list,)
#print(model_1)
#print("MODEL 1 \n\n")
#summary(model_1,input_size=(16,32,32),device="cpu")
#config_list2=[(1,32,64)]
#model_2 = AUX_Model_TOFD(config_list=config_list2,)
#print("MODEL 2 \n\n")
#summary(model_2,input_size=(32,16,16),device="cpu")
#model_3 = AUX_Model_TOFD(config_list= [])
#print("MODEL 3 \n\n")
#summary(model_3,input_size=(64,8,8),device="cpu")





#from models.OOG_resnet import ResNet34


#res34 = ResNet34(num_classes=100)

#res34_outputs = res34(virtual_input)

#summary(res34,input_size=(3,32,32),device="cpu")


#for out_34 in res34_outputs:
    #print("ResNet 34 outputs ===> ",out_34.shape)



#config_list=[(1,64,128),(1,128,256),(1,256,512)]
#config_list=[(64,128),(128,256),(256,512)]
#config_list_wres28_2=[(32,128),(64,256),(128,512)]
                 #(1,32,64),
                 #(0,0,0),]
#config_list = config_list_wres28_2


#model_1 = AUX_Model_TOFD(config_list=config_list,expansion=1,coeeficient=512,num_classes=200)
#print(model_1)
#print("MODEL 1 \n\n")
#summary(model_1,input_size=(64, 32, 32),device="cpu")
#""""""