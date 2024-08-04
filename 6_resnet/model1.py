import torch
import torch.nn as nn 
import torch.nn.functional as F 


# 18, 34 和其他更高层的Residual Block结构是不一样的
# this is for 18, 34 layers
class BasicBlock(nn.Module):
    expansion = 1 # 每一个残差块内部卷积核输出channels数是不是一样的，1代表一样
    
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    
    def forward(self, x):
        identity = x 
        if self.down_sample:
            identity = self.down_sample(x) # might need to convert the size to make them addable
        x = self.conv1(x) # size reduced to half 
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = torch.add(identity, x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, 
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        if self.down_sample:
            identity = self.down_sample(x) 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        out = torch.add(x, identity)
        out = self.relu(out)
        return out 

class ResNet(nn.Module):
    def __init__(self, block, blocks_params, num_classes=1000, include_top=True):
        """init function for ResNet

        Args:
            block (class): what block to use, 18 and 34 layer: BasicBlock; higher-layer: Bottleneck
            blocks_params (list): parameter list, how many blocks in each layer
            num_classes (int, optional): how many classes to classify. Defaults to 1000.
            include_top (bool, optional): whether to include the first two layers (a conv and a pool). Defaults to True.
        """
        super().__init__()
        self.include_top = include_top
        self.in_channels = 64 # 经过top层的处理后是64通道，记录每一个layer输入通道是多少 
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # 112 * 112
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 56 * 56 
        
        self.layer1 = self._make_layer(block, 64, blocks_params[0]) 
        self.layer2 = self._make_layer(block, 128, blocks_params[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, blocks_params[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_params[3], stride=2)
        
    def _make_layer(self, block, channel, block_param, stride=1):
        """_summary_

        Args:
            block (class): BasicBlock or Bottleneck
            channel (int): out_channels of the first Conv2d in this layer
            block_param (list): _description_
            stride (int, optional): stride. Defaults to 1.
        """
        down_sample = None # may need to adjust the HW or out_channels 
        if stride != 1 or self.in_channels != channel * block.expansion: # why????
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channel, down_sample)) # 每一层的第一个残差块可能需要下采样处理 
        self.in_channels = channel * block.expansion
        for _ in range(1, block_param):
            layers.append(block(self.in_channels, channel))
        
        return nn.Sequential(*layers)
            
        
        
        
        
    
        
        