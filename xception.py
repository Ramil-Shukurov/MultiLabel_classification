#Xception model

import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                               groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class middleflow(nn.Module):
    def __init__(self) -> None:
        super(middleflow, self).__init__()

        self.layer = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(728,728, 3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728,728, 3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728,728, 3),
            nn.BatchNorm2d(728),
        )

    def forward(self, x):
        identity = x
        x =  self.layer(x)
        x = x + identity
        return x

class exitflow(nn.Module):
    def __init__(self) -> None:
        super(exitflow, self).__init__()

        self.layer = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(728,728,3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728,728,3),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 2,1)
        )

        self.conv1 = nn.Conv2d(728,728,1,2, 0)

        self.layer2 = nn.Sequential(
            SeparableConv2d(728,1536,3),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeparableConv2d(1536,2048,3),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
    
    def forward(self, x):
        con1 = self.conv1(x)
        x = self.layer(x)
        x = x + con1
        x = self.layer2(x)
        return x

class Xception(nn.Module):
    def __init__(self, in_channels=3, num_classes = 12) -> None:
        super(Xception, self).__init__()
        
        # entry field - begins
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64,128,1,2,0),
            nn.BatchNorm2d(128))
        
        self.layer2 = nn.Sequential(
            SeparableConv2d(64,128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeparableConv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer3 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeparableConv2d(256,256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,256,1,2, 0),
            nn.BatchNorm2d(256))

        self.layer4 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(256, 728, kernel_size=3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728,728, kernel_size=3),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 2, 1))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256,728,1,2, 0),
            nn.BatchNorm2d(728))
        # entry field - ends

        self.middle = middleflow()
        self.exitf = exitflow()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        con, layer = self.conv1(x), self.layer2(x)
        x =  con + layer
        con, layer =  self.conv2(x), self.layer3(x)
        x = con + layer 
        con, layer = self.conv3(x), self.layer4(x)
        x = con + layer
        
        x = self.middle(x)
        x = self.middle(x)
        x = self.middle(x)
        x = self.middle(x)
        x = self.middle(x)
        x = self.middle(x)
        x = self.middle(x)
        x = self.middle(x)

        x = self.exitf(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
