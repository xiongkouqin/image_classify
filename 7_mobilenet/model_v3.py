from functools import partial
from typing import Callable, List, Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F 


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10% 
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

def adjust_channel(channels: int, width_multi: float):
    return _make_divisible(width_multi * channels, 8)

class ConvBNActivation(nn.Sequential):
    
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        
class SqueezeExcitation(nn.Module):
    
    def __init__(self, input_c:int, squeeze_factor: int = 4) -> None:
        super().__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # self.fc1 = nn.Linear(input_c, squeeze_c)
        # self.fc2 = nn.Linear(squeeze_c, input_c)
        # 这里不用全连接层，用1*1卷积层, 我觉得是因为shape方便 
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True) # batch * channels * 1 * 1 , x should be batch * H * W, broadcasting here
        return scale * x 
    
    
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float) -> None:
        """Parameters for Bneck, Archiecture is as followed:
        - 1*1 Conv2d: to lift up the dimensions to expanded_c * input_c
        - D-wise Conv: stride, kernel_size is for this layer
        - SE block? optional
        - 1*1 Conv2d: to adjust the dimension to output_c 
        Args:
            input_c (int): the number channels of input 
            kernel (int): kernel_size of the dw conv layer
            expanded_c (int): first 1x1 layer should lift the channnel to (expanded_c * input_c)
            out_c (int): output_channel 
            use_se (bool): use se block or not 
            activation (str): relu or hardswish 
            stride (int): stride for DW
            width_multi (float):  alpha 
        """
        self.input_c = adjust_channel(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = adjust_channel(expanded_c, width_multi)
        self.out_c = adjust_channel(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride 
    
    # def adjust_channel(channels: int, width_multi: float):
    #     return _make_divisible(width_multi * channels, 8)
    
    
class InvertetdResidual(nn.Module):
    
    def __init__(self, 
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]) -> None:
        super().__init__()
        
        if cnf.stride not in [1, 2]:
            raise ValueError('illegal stride value!')
        
        self.use_res_connect = cnf.stride == 1 and cnf.input_c == cnf.out_c
        
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        
        # expand, whether to expand 
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(
                cnf.input_c, cnf.expanded_c, 
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer))
            
        # depth-wise
        layers.append(ConvBNActivation(
            cnf.expanded_c, cnf.expanded_c,kernel_size=cnf.kernel,stride=cnf.stride, groups=cnf.expanded_c,
            norm_layer=norm_layer, activation_layer=activation_layer
        ))
        
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))
        
        # to change the dim, 其实这里我记得论文里也没有activation？ wok所以用Identity
        layers.append(ConvBNActivation(
            cnf.expanded_c, cnf.out_c, kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity
        ))
        
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1 
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x 
        return result

class MobileNetV3(nn.Module):
    
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,  # 倒数第二个全连阶层（卷积层实现）输出节点个数
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer:Optional[Callable[..., nn.Module]] = None,) -> None:
        super().__init__()
        
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not ((isinstance(inverted_residual_setting, List)) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        
        if block is None:
            block = InvertetdResidual
            
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        
        layers: List[nn.Module] = []
        
        # build the first conv layer
        first_conv_ouput_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3, first_conv_ouput_c, 3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        
        # build the bnecks 
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        
        # build the last several layers
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = 6 * last_conv_input_c
        layers.append(ConvBNActivation(last_conv_input_c, last_conv_output_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_output_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
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
                
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x 
    
    
def mobile_net_v3_large(num_classes:int = 1000, reduced_tail:bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0 
    bneck_cnf = partial(InvertedResidualConfig, width_multi=width_multi) # so I dont need to pass the parameters everytime calling it
    reduce_divider = 2 if reduced_tail else 1
    inverted_residual_setting = [
        bneck_cnf(16, 3, 16, 16, False, 'RE', 1),
        bneck_cnf(16, 3, 64, 24, False, 'RE', 2), # C1
        bneck_cnf(24, 3, 72, 24, False, 'RE', 1), 
        bneck_cnf(24, 5, 72, 40, True, 'RE', 2), #C2
        bneck_cnf(40, 5, 120, 40, True, 'RE', 1),
        bneck_cnf(40, 5, 120, 40, True, 'RE', 1),
        bneck_cnf(40, 3, 240, 80, False, 'HS', 2), # C3
        bneck_cnf(80, 3, 200, 80, False, 'HS', 1),
        bneck_cnf(80, 3, 184, 80, False, 'HS', 1),
        bneck_cnf(80, 3, 184, 80, False, 'HS', 1),
        bneck_cnf(80, 3, 480, 112, True, 'HS', 1),
        bneck_cnf(112, 3, 672, 112, True, 'HS', 1),
        bneck_cnf(112, 5, 672, 160 // reduce_divider, True, 'HS', 2), #C4
        bneck_cnf(160 // reduce_divider, 5, 960, 160 // reduce_divider, True, 'HS', 1),
        bneck_cnf(160 // reduce_divider, 5, 960, 160 // reduce_divider, True, 'HS', 1),
    ]
    
    last_channel = adjust_channel(1280 // reduce_divider, width_multi) # C5
    
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting, 
                       last_channel=last_channel,
                       num_classes=num_classes)

def mobile_net_v2_large(w) -> MobileNetV3:
    pass  


         
        