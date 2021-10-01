import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torchsummary import summary


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



#reproducible_state()
def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_TOFD_CIFAR(nn.Module):

    def __init__(self, block, layers, num_classes=10,zero_init_residual=False, align="CONV"):
        super(ResNet_TOFD_CIFAR, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        if num_classes == 200:
            # for Tiny Image_Net

            self.avgpool = nn.AvgPool2d(16, stride=1)
        else:
            # for Cifar10-100
            self.avgpool = nn.AvgPool2d(8, stride=1)

        self.auxiliary1 = nn.Sequential(
            SepConv(
                channel_in=16 * block.expansion,
                channel_out=32 * block.expansion
            ),
            SepConv(
                channel_in=32 * block.expansion,
                channel_out=64 * block.expansion
            ),
            #SepConv(
             #   channel_in=32 * block.expansion,
            #    channel_out=64 * block.expansion
           # ),
            #nn.AvgPool2d(4, 4)
            #nn.AvgPool2d(8, stride=1)
            self.avgpool
        )

        self.auxiliary2 = nn.Sequential(
           # SepConv(
             #   channel_in=32 * block.expansion,
             #   channel_out=64 * block.expansion,
            #),
            SepConv(
                channel_in=32 * block.expansion,
                channel_out=64 * block.expansion,
            ),
            #nn.AvgPool2d(4, 4)
            #nn.AvgPool2d(8, stride=1)
            self.avgpool
        )
        self.auxiliary3 = nn.Sequential(
            self.avgpool

        )


        self.fc1 = nn.Linear(64 * block.expansion, num_classes)
        self.fc2 = nn.Linear(64 * block.expansion, num_classes)
        self.fc3 = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)













        #if num_classes == 200:
            # for Tiny Image_Net

            #self.avgpool = nn.AvgPool2d(16, stride=1)
        #else:
            # for Cifar10-100
            #self.avgpool = nn.AvgPool2d(8, stride=1)


        self.fc = nn.Linear(64 * block.expansion, num_classes)


        #for m in self.modules():
            #if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
               # m.weight.data.normal_(0, math.sqrt(2. / n))
           # elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        layer1_output = self.layer1(x)
        feature_list.append(layer1_output)
        layer2_output = self.layer2(layer1_output)
        feature_list.append(layer2_output)
        layer3_output = self.layer3(layer2_output)

        feature_list.append(layer3_output)

        out1_feature = self.auxiliary1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.auxiliary2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(feature_list[2]).view(x.size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)

        x = self.avgpool(layer3_output)
        #print("AVG POOL  ===> ", x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)




        return [x,out3, out2, out1], [ out3_feature, out2_feature, out1_feature]




def resnet8_cifar(seed = 3,**kwargs):
    reproducible_state(seed=seed)
    model = ResNet_TOFD_CIFAR(BasicBlock, [1, 1, 1], **kwargs)
    return model



def resnet14_cifar(seed = 3,**kwargs):
    reproducible_state(seed=seed)
    model = ResNet_TOFD_CIFAR(BasicBlock, [2, 2, 2], **kwargs)
    return model


def resnet20_cifar(seed = 3,**kwargs):
    reproducible_state(seed=seed)
    model = ResNet_TOFD_CIFAR(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(seed=3,**kwargs):
    reproducible_state(seed=seed)
    model = ResNet_TOFD_CIFAR(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar(seed=3,**kwargs):
    reproducible_state(seed=seed)
    model = ResNet_TOFD_CIFAR(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(seed=3,**kwargs):
    reproducible_state(seed=seed)
    model = ResNet_TOFD_CIFAR(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(seed=3,**kwargs):
    reproducible_state(seed=seed)
    model = ResNet_TOFD_CIFAR(BasicBlock, [18, 18, 18], **kwargs)
    return model



#test =resnet8_cifar(seed=3,num_classes=200)
#summary(test,input_size=(3,64,64),device="cpu")