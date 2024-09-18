import torch
import torch.nn as nn

class block(nn.Module):
    """
    Initialize a ResNet block

    Args:
        in_channels (_type_): # of channels going into this block
        out_channels (_type_): 4*out_channels = # of channels coming out of this block, 
                               out_channels = # of channels for middle conv layers
        identity_downsample (PyTorch layer, optional): if skip connection requires change of dimension
        stride (int, optional): stride of middle Conv2D layer
    """
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity   # residual learning
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        """
        ResNet implementation

        Args:
            block (custom type): a block of residual network
            layers (list): list of # of blocks in each layer
            image_channels (_type_): # of channels in the input image
            num_classes (_type_): # of output classes
        """

        super(ResNet, self).__init__()

        self.in_channels = 64
        # original image pixel size = 3 * 224 * 224, all numbers will based off of this size

        # padding in the LOC below is inferred from original paper, as dimension of image goes from 3 * 224 * 224 -> 3 * 112 * 112
        # with stride of 2 and kernel size of 7, halving the size requires 3 pixels of padding
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)  # output = 64 * 112 * 112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                 # output = 64 * 56 * 56

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)     # output = 256 * 56 * 56
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)    # output = 512 * 28 * 28
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)    # output = 1024 * 14 * 14
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)    # output = 2048 * 7 * 7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                                     # output = 2048 * 1 * 1
        # reshape in between in forward()                                               # output = 2048
        self.fc = nn.Linear(2048, num_classes)                                          # output = num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        '''
        Explanation for LOC of condition below:

        Case 1: stride != 1
            e.g. layer looks like this:
                block 1:
                    input = 512 * 28 * 28 (channels * pixels * pixels)
                    conv1: 256 * 28 * 28 (first conv always stride = 1)
                    conv2: 256 * 14 * 14 (middle conv stride = 2)
                    conv3: 1024 * 14 * 14
                block 2:
                    input = 1024 * 14 * 14
                    conv1: 256 * 14 * 14
                    conv2: 256 * 14 * 14
                    conv3: 1024 * 14 * 14
                block 3:
                    ...
            identity_downsample needed for block 1 since skip connection is from 
                before conv1 -> conv3
        
        Case 2: self.in_channels != out_channels * 4
            this is a special case for the first block of the first layer
            for block 1 -> 2, 2 -> 3, etc., input is 256 * 56 * 56 -> 64 * 56 * 56
            but for first maxpool layer -> layer 1 block 1, input is 64 * 56 * 56
            so this identity_downsample isn't to resize pixels, but just a 
            Conv2d + BN to match # of channels
        '''
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4     # num channels goes something like 4n, n, ..., n, 4n

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)

'''
def test():
    net = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)

test()
'''