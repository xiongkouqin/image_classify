import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchinfo import summary


# Residual Block for ResNet-18/34
class BasicBlock(nn.Module):
    # in a residual block, there are convs
    # expansion = # of channels of the last conv / # of channels of other convs
    expansion = 1 
    
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """How to initialize a Residual Block for ResNet-18/34
        There are just two convs, for the first one, it will change the channels and HW
        Also, the stride is for the first conv, 
        because for the first residual block, it will not change the HW (stride=1)
        if stride = 2, means the block will reduce HW to 1/2
        Args:
            in_channels (int): num_channels of input for the whole block
            out_channels (int): num_channels of ouput for the whole block
            stride (int): stride for the first conv 
            downsample (nn.Module): whether a downsample needs to be applied to X, to make the channels and HW match 
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x 
        
        if self.downsample:
            identity = self.downsample(identity)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        out = torch.add(x, identity)
        
        return self.relu(out)
    
    
class BotteleNeck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x 
        
        if self.downsample:
            identity = self.downsample(identity)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        out = torch.add(x, identity)
        
        return self.relu(out)


class ResNet(nn.Module):
    
    def __init__(self,block: type, block_num_list:list, num_classes=1000, include_top=True):
        """Construct ResNet 

        Args:
            block (type): what type of the residual block will be used BasicBlock or BottleNeck 
            block_num_list (list): in each layer, how many blocks
            num_classes (int, optional): number of classes to predict . Defaults to 1000.
            include_top (bool, optional): whether include the non residual blocks. Defaults to True.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.cur_in_channels = 64 
        self.layer1 = self._make_layer(block, block_num_list[0], 64)
        self.layer2 = self._make_layer(block, block_num_list[1], 128, 2)
        self.layer3 = self._make_layer(block, block_num_list[2], 256, 2)
        self.layer4 = self._make_layer(block, block_num_list[3], 512, 2)
        
        self.include_top = include_top
        if include_top:
            self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.cur_in_channels, num_classes)
    
    def _make_layer(self, block, block_num, out_channels, stride=1):
        # first, check whether a downsample is needed?
        # 1. stride != 1 means the HW maintain the same
        # 2. self.in_channels != out_channels * block.expansion 
        layer = []
        for i in range(0, block_num):
            if i != 0:
                stride = 1
            downsample = None 
            if stride != 1 or self.cur_in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.cur_in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion)
                )
            blk = block(self.cur_in_channels, out_channels, stride, downsample)
            layer.append(blk)
            self.cur_in_channels = out_channels * block.expansion
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
        return x 


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, include_top)

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top)

def resnet50(num_classes=1000, include_top=True):
    return ResNet(BotteleNeck, [3, 4, 6, 3], num_classes, include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(BotteleNeck, [3, 4, 23, 3], num_classes, include_top)

def resnet152(num_classes=1000, include_top=True):
    return ResNet(BotteleNeck, [3, 8, 36, 3], num_classes, include_top)

if __name__ == '__main__':
    # input = torch.randn((17, 3, 224, 224))
    # net1 = resnet18()
    # net2 = resnet34()
    # net3 = resnet50()
    # net4 = resnet101()
    # net5 = resnet152()
    # print(net1(input).shape)
    # print(net2(input).shape)
    # print(net3(input).shape)
    # print(net4(input).shape)
    # print(net5(input).shape)
    
    # Print a summary using torchinfo (uncomment for actual output)
    model = resnet34()
    summary(
        model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ) 
