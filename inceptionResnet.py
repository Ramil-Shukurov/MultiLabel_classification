# Inception ResnetV1 

import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_block,self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class block_A(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(block_A, self).__init__()
        self.branch1 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.conv1 = conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0) 
        self.conv2 = nn.Conv2d(in_channels=out_channels*3, out_channels=in_channels, kernel_size=1, stride=1, padding=0) 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = torch.cat([self.branch1(x), self.branch2(x), self.conv1(x)], 1)
        x = self.conv2(x)
        x = identity + x
        x = self.relu(x)
        return x

class reduction_A(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(reduction_A, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=k, kernel_size=1, stride=1, padding=0),
            conv_block(in_channels=k, out_channels=l, kernel_size=3, stride=1, padding=1),
            conv_block(in_channels=l, out_channels=m, kernel_size=3, stride=2, padding=1)
        )
        self.conv3 = conv_block(in_channels=in_channels, out_channels=n, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.conv3(x), self.pool(x)], 1)
        #x = self.relu(x)
        return x

class block_B(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(block_B, self).__init__()

        self.branch = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(7,1), stride=1, padding=(3,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_channels*2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = torch.cat([self.branch(x), self.conv1(x)], 1)
        x = self.conv2(x)
        x = identity + x
        x = self.relu(x)
        return x

class reduction_B(nn.Module):
    def __init__(self, in_channels, out_channels, out_channels2):
        super(reduction_B, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, out_channels, 1,1,0),
            conv_block(out_channels,out_channels,3,1,1),
            conv_block(out_channels, out_channels, 3,2,1)
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, out_channels,1,1,0),
            conv_block(out_channels, out_channels,3,2,1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, out_channels,1,1,0),
            conv_block(out_channels, out_channels2,3,2,1)
        )

        self.pool = nn.MaxPool2d(3,2,1)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.pool(x)], 1)
        #x = self.relu(x)
        return x

class block_C(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(block_C, self).__init__()

        self.branch = nn.Sequential(
            conv_block(in_channels, out_channels, 1,1,0),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = conv_block(in_channels, out_channels,1,1,0)
        self.conv2 = nn.Conv2d(out_channels*2, in_channels,1,1,0)
        #self.downsample = nn.Conv2d(in_channels, out_channels2,1,1,0)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x#self.downsample(x)
        x = torch.cat([self.branch(x), self.conv1(x)],1)
        x = self.conv2(x)
        x = identity + x
        x = self.relu(x)
        return x

class InceptionResnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=12):
        super(InceptionResnet, self).__init__()
        # stem begins
        self.stem = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
            conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv_block(in_channels=64, out_channels=80, kernel_size=1, stride=1, padding=1),
            conv_block(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1),
            conv_block(in_channels=192, out_channels=256, kernel_size=3, stride=2, padding=1),
        )
        # stem ends

        self.inceptionresnetA1 = block_A(256, 32)
        self.inceptionresnetA2 = block_A(256, 32)
        self.inceptionresnetA3 = block_A(256, 32)
        self.inceptionresnetA4 = block_A(256, 32)
        self.inceptionresnetA5 = block_A(256, 32)

        self.reductiona = reduction_A(256, 192,192,256,384)

        self.inceptionresnetB1 = block_B(896, 128)
        self.inceptionresnetB2 = block_B(896, 128)
        self.inceptionresnetB3 = block_B(896, 128)
        self.inceptionresnetB4 = block_B(896, 128)
        self.inceptionresnetB5 = block_B(896, 128)
        self.inceptionresnetB6 = block_B(896, 128)
        self.inceptionresnetB7 = block_B(896, 128)
        self.inceptionresnetB8 = block_B(896, 128)
        self.inceptionresnetB9 = block_B(896, 128)
        self.inceptionresnetB10 = block_B(896, 128)

        self.reductionb = reduction_B(896, 256, 384)

        self.inceptionresnetC1 = block_C(1792,192)
        self.inceptionresnetC2 = block_C(1792,192)
        self.inceptionresnetC3 = block_C(1792,192)
        self.inceptionresnetC4 = block_C(1792,192)
        self.inceptionresnetC5 = block_C(1792,192)

        self.avgpool = nn.AvgPool2d(7,1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1792,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num_classes)
        )
        
    
    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionresnetA1(x)
        x = self.inceptionresnetA2(x)
        x = self.inceptionresnetA3(x)
        x = self.inceptionresnetA4(x)
        x = self.inceptionresnetA5(x)
        
        x = self.reductiona(x)

        x = self.inceptionresnetB1(x)
        x = self.inceptionresnetB2(x)
        x = self.inceptionresnetB3(x)
        x = self.inceptionresnetB4(x)
        x = self.inceptionresnetB5(x)
        x = self.inceptionresnetB6(x)
        x = self.inceptionresnetB7(x)
        x = self.inceptionresnetB8(x)
        x = self.inceptionresnetB9(x)
        x = self.inceptionresnetB10(x)

        x = self.reductionb(x)

        x = self.inceptionresnetC1(x)
        x = self.inceptionresnetC2(x)
        x = self.inceptionresnetC3(x)
        x = self.inceptionresnetC4(x)
        x = self.inceptionresnetC5(x)

        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x
