import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True):
        super().__init__()
        self.aux_logits = aux_logits
        
        # input 224 x 224 x 3
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3) # 112 x 112 x 64  (224 - 7 + 2p) / 2 + 1 = 112 -> p = 2.5 
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # 56 x 56 x 64, (112 - 3 + 2p) / 2 + 1 = 56, p = 0.5 or I can use ceil mode 
        # skip local response normalization
        self.conv2 = BasicConv2d(64, 192,kernel_size=1) # 56 * 56 * 192 (56 - 1 + 2p) / 1 + 1 = 56, padding = 0
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # 28 x 28 x 192
        
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32) # 28 x 28 x 256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64) # 28 x 28 x 480
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # 14 x 14 x 480
        
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64) # 14 x 14 x 512
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64) # 14 x 14 x 512
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64) # 14 x 14 x 512
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64) # 14 x 14 x 528
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128) # 14 x 14 x 832
        
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # 7 x 7 x 832
        
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128) # 7 x 7 x 832
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128) # 7 x 7 x 1024
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes=num_classes)
            self.aux2 = InceptionAux(528, num_classes=num_classes)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1) # 1 * 1 * 1024 I can also use AdaptiveAvgPool which will always convert to [1, 1]
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avg_pool(x)
        x = self.fc(self.dropout(self.flatten(x)))
        
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x         
        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
             
        
        
         
        

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1) # (in - k + 2p) / s + 1, padding here is to keep same HW
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
        
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x) 
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        res = torch.cat(outputs, dim=1)   
        return res      

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        # output size
        # (14 - 5) / 3 + 1 = 4 
        # 4a 14 * 14 * 512  -> 4 * 4 * 512 -> 4 * 4 * 128
        # 4d 14 * 14 * 528  -> 4 * 4 * 528 -> 4 * 4 * 128
        self.fc1 = nn.Linear(4*4*128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1) # keep the dim 0, batch 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x 

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x 


if __name__ == '__main__':
    net = GoogLeNet()
    input = torch.randn((28, 3, 224, 224))
    output, aux1, aux2 = net(input) # check here is a tuple
    print(output.shape)