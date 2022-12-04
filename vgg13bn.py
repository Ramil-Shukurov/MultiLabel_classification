    # VGG13 + BN

import torch.nn
class VGG13(torch.nn.Module):
    def __init__(self, num_classes = 12):
        super(VGG13, self).__init__()

        self.block_1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                       stride=(2, 2))
        )
        self.block_2 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                       stride=(2, 2))
        )

        self.block_3 = torch.nn.Sequential(        
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                       stride=(2, 2))
        )
        self.block_4 = torch.nn.Sequential(   
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.ReLU(),        
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.ReLU(),      
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                       stride=(2, 2))
        )
        self.block_5 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.ReLU(),    
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                       stride=(2, 2))
        )
        height, width = 6, 6 ## you may want to change that depending on the input image size
        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512*height*width, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4096, 1000),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.classifier(x)
        #probas = F.softmax(logits, dim=1)
        return x  
