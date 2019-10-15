import torch
import torch.nn as nn


def vgg_block(num_convs, in_channels, num_channels):
    layers=[]
    for i in range(num_convs):
        layers+=[nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1)]
        in_channels=num_channels
    layers +=[nn.ReLU()]
    layers +=[nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)
    
class vgg16(nn.Module):
    def __init__(self, num_classes = 10):
        super(vgg16,self).__init__()
        self.conv_arch=((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2,512,512))
        layers=[]
        for (num_convs,in_channels,num_channels) in self.conv_arch:
            layers+=[vgg_block(num_convs,in_channels,num_channels)]
        self.features=nn.Sequential(*layers)
        self.dense1 = nn.Linear(512*7*7,4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096, num_classes)

    def forward(self,x):
        x=self.features(x)
        x=x.view(-1,512*7*7)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x
