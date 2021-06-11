import torch.nn as nn
import torch.nn.functional as F
import torch
from efficientnet_pytorch import EfficientNet

import os
__all__ = ['ResNet50', 'ResNet101','ResNet152', 'VGG19','EfficientB7']

vgg_config = {
    'VGG11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    'VGG19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 125, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']
}
efficientnet_config = {
    'b0':'efficientnet-b0',
    'b1':'efficientnet-b1',
    'b2':'efficientnet-b2',
    'b3':'efficientnet-b3',
    'b4':'efficientnet-b4',
    'b5':'efficientnet-b5',
    'b6':'efficientnet-b6',
    'b7':'efficientnet-b7',
}

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, expansion = 4, init = True, pre_train=True,feature_dim=128):
        super(ResNet,self).__init__()
        self.expansion = expansion
        self.init = init
        self.pre_train = pre_train
        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        #MLP layers
        self.g = nn.Sequential(nn.Linearbi(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        if self.init == True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.pre_train == False:
            return x
        features = torch.flatten(x,start_dim=1) #shape = [B, 2048]
        x = self.g(features) #[B,128]
        return F.normalize(features, dim=-1), F.normalize(x, dim=-1)

class VGG(nn.Module):
    def __init__(self,vgg_config = vgg_config, model='VGG19',init = True,class_num=5):
        super(VGG, self).__init__()
        self.config = vgg_config
        self.skeleton = self.make_layers(config=model,init_channel=3)
        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.classifier = nn.Linear(512,class_num,bias=False)
        self.init = init
        if self.init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.skeleton(x)
        x = self.avgpool(x)
        feature = torch.flatten(x,start_dim=1)
        output = self.classifier(feature)
        return output

    def make_layers(self,config, init_channel=3):
        layers = []
        in_channel = init_channel
        for layer in self.config[config]:
            if layer == 'P':
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=in_channel,out_channels=layer,kernel_size=3,padding=1))
                layers.append(nn.BatchNorm2d(layer))
                layers.append(nn.ReLU(inplace=True))
                in_channel = layer
        return nn.Sequential(*layers)

class Efficientnet(nn.Module):
    def __init__(self,config = 'b7',class_num=5):
        super(Efficientnet,self).__init__()
        self.skeleton = EfficientNet.from_pretrained(efficientnet_config[config])
        feature = self.skeleton._fc.in_features
        self.skeleton._fc = nn.Linear(in_features=feature,out_features=class_num,bias=False)

    def forward(self,x):
        return self.skeleton(x)

class Linear(nn.Module):
    def __init__(self,model_name,num_class=5,init=False, pre_train=False,pre_load=True,pretrained_path="checkpoint"):
        super(Linear,self).__init__()
        if model_name in ['resnet101', 'lbp_resnet101']:
            self.net = ResNet101(init=init,pre_train=pre_train)
        elif model_name == 'resnet50':
            self.net = ResNet50(init=init,pre_train=pre_train)
        self.pre_load=pre_load
        if self.pre_load:
            assert pretrained_path in ['checkpoint.pth.tar','model_best.pth.tar'],\
                'invalid pretrained path %s'%(pretrained_path)
            path = os.path.join("checkpoint","self_supervised",model_name,pretrained_path)
            print("Loading pre_trained model %s"%(path))
            self.load_state_dict(torch.load(path,map_location='cpu')['state_dict'],strict=False)
            # self.load_state_dict(
            #     {'net.'+ k : v for k, v in torch.load(path,map_location='cpu')['state_dict'].items()}
            # )
        self.fc = nn.Linear(2048,num_class,bias=False)

    def forward(self,x):
        x = self.net(x)
        feature = torch.flatten(x,start_dim=1)
        out = self.fc(feature)
        return out

def ResNet50(init = True, pre_train=True,feature_dim=128,):
    return ResNet([3, 4, 6, 3], init = init, pre_train=pre_train,feature_dim=feature_dim)

def ResNet101(init = True, pre_train=True,feature_dim=128):
    return ResNet([3, 4, 23, 3], init = init, pre_train=pre_train,feature_dim=feature_dim)

def ResNet152(init = True, pre_train=True,feature_dim=128):
    return ResNet([3, 8, 36, 3], init = init, pre_train=pre_train,feature_dim=feature_dim)

def VGG11(init = True, class_num = 5):
    return VGG(model='VGG11',init=init,class_num=class_num)

def VGG13(init = True, class_num = 5):
    return VGG(model='VGG13',init=init,class_num=class_num)

def VGG16(init = True, class_num = 5):
    return VGG(model='VGG16',init=init,class_num=class_num)

def VGG19(init = True, class_num = 5):
    return VGG(model='VGG19',init=init,class_num=class_num)

def EfficientB0():
    return Efficientnet('b0',class_num=5)

def EfficientB1():
    return Efficientnet('b1',class_num=5)

def EfficientB2():
    return Efficientnet('b2',class_num=5)

def EfficientB3():
    return Efficientnet('b3',class_num=5)

def EfficientB4():
    return Efficientnet('b4',class_num=5)

def EfficientB5():
    return Efficientnet('b5',class_num=5)

def EfficientB6():
    return Efficientnet('b6',class_num=5)

def EfficientB7():
    return Efficientnet('b7',class_num=5)