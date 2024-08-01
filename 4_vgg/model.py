import torch 
import torch.nn as nn 


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = features # 直接传过来
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M': # MaxPool 
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else: # Conv2d, hey dont forget ReLU 
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1, stride=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v 
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name='vgg16', **kwargs):
    assert model_name in cfgs, f'Warning: model number {model_name} not in cfgs dict!'
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    return model 
