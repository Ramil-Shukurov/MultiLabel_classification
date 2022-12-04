# Alexnet model
import torch.nn
class Alexnet(torch.nn.Module):
    def __init__(self, in_channels = 3, num_classes = 12):
        super(Alexnet, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=64,
                            kernel_size=(11, 11),
                            stride=(4, 4),
                            padding=2),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size=(3, 3),
                                stride=(2, 2))
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=192,
                            kernel_size=(5, 5),
                            stride=(1, 1),
                            padding=2),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size=(3, 3),
                                stride=(2, 2))
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=192,
                            out_channels=384,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels=384,
                            out_channels=256,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=256,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3),
                                stride=(2, 2))
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((6,6))

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256*6*6,4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(4096,num_classes),    
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
