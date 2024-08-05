import torch 
import torch.nn as nn
import math


def _make_divisible(ch, divisor=8, min_ch=None):
    """Compute new_ch(channel), that is divisible by divisor and return the nearest one  

    Args:
        ch (_type_): the original channel 
        divisor (int, optional): _description_. Defaults to 8.
        min_ch (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10% 
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLU(nn.Sequential):
    
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        """Create a module contains: Conv2d, BatchNorm and ReLU6 as it repeats a lot

        Args:
            in_channel (_type_): _description_
            out_channel (_type_): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
            stride (int, optional): _description_. Defaults to 1.
            groups (int, optional): if it's equals to 1, a common conv, if = in_channel, DW Conv . Defaults to 1.
        """
        # TODO check for DW, because the in_channel = out_channel = groups should be satisified for DW 
        # out = (in - k + 2p ) / s + 1 = in / s 
        # in/s - k/s + 2p/s + 1 = in/s
        # 2p/s = k/s - 1
        # p = (k - s) / 2
        padding = math.ceil((kernel_size - stride) // 2) # different from author, let's try 
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )
        
class InvertedResidual(nn.Module):
    
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super().__init__()
        self.use_shortcut = in_channel == out_channel and stride == 1
        layers = []
        
        # One InvertedResidual has 3 components:
        # - 1x1 Conv, expand the dim to k times (so if expand_ratio = 1, it can be skipped)
        # - DW 
        # - PW with linear activation  
        hidden_channel = in_channel * expand_ratio
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))  
        layers.extend([
            # 3x3 depth-wise
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1*1 pixel-wise (linear, dont use activation)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])  
        self.conv = nn.Sequential(*layers)
        
    
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
        
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super().__init__()
        block = InvertedResidual
        # 减少一些通道数来减少计算量
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)
        
        inverted_residual_setting = [
            # t: expand_ratio
            # c: output_channel
            # n: how many residuals
            # s: stride, for the first invertible residual in the layer, others stride = 1
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        features = []
        
        # conv1 layers
        features.append(ConvBNReLU(3, input_channel, stride=2))
        
        for t, c, n , s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel,stride, expand_ratio=t))
                input_channel = output_channel
                
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)
        
        # building the classfier
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        # original it uses Conv2d 1x1, but it's equalvaleint to fc
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
                
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x 
        

        
if __name__ == '__main__':
    from torchinfo import summary
    model = MobileNetV2(num_classes=5)
    x = torch.randn((32, 3, 224, 224))
    print(model(x))
    # summary(
    #     model=model, 
    #     input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
    #     # col_names=["input_size"], # uncomment for smaller output
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"]
    # ) 