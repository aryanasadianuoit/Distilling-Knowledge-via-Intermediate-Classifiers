from torch import nn




cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self,vgg_name,num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class VGG_Intermediate_Branches(nn.Module):
    def __init__(self,vgg_name,num_classes=10):
        super(VGG_Intermediate_Branches, self).__init__()
        self.vgg_name= vgg_name
        features = self._make_layers(cfg[vgg_name])
        if self.vgg_name == "VGG16":
            self.features_1 = features[:3]
            self.features_2 = features[3:6]
            self.features_3 = features[6:10]
            self.features_4 = features[10:14]
            self.features_5 = features[14:]
        else:
            self.features_1 = features[:4]
            self.features_2 = features[4:7]
            self.features_3 = features[7:10]
            self.features_4 = features[10:]

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        intermediate_outputs_list = []
        out_1 = self.features_1(x)
        intermediate_outputs_list.append(out_1)
        out_2 = self.features_2(out_1)
        intermediate_outputs_list.append(out_2)

        out_3 = self.features_3(out_2)
        intermediate_outputs_list.append(out_3)

        out_4 = self.features_4(out_3)
        intermediate_outputs_list.append(out_4)

        if self.vgg_name  == "VGG16":
            out_5 = self.features_5(out_4)
            out = out_5.view(out_5.size(0), -1)
            intermediate_outputs_list.append(out_5)
        else:
            out = out_4.view(out_4.size(0), -1)
        out = self.classifier(out)
        return out, intermediate_outputs_list

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
