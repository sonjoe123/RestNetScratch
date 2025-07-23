import  torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,input_embed, output_embed, stride, bias = False ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_embed,output_embed,kernel_size=3 ,stride= stride, padding = 1, bias= bias),
            nn.BatchNorm2d(output_embed),
        )
        # Second 3x3 conv: always uses stride=1 to preserve spatial size
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_embed, output_embed,kernel_size=3, stride= 1, padding= 1, bias = bias),
            nn.BatchNorm2d(output_embed),
        )
        self.ReLU = nn.ReLU()

        # In case the shape of x is not matched with f(x) we need to downsample x itself so that it match - basically a cnn layer for x
        # First we have to set it to None so that it does not automatically get downsample
        self.downsample = None
        # Downsample when: input_embed != output_embed (input_embed is basically x dimension here)
        # or when stride !=1 this is because if stride is not 1 it means that there have been a downsample happening and we need to fix the x for that
        # kernel_size = 1 since this is a 1x1 Convolution because we only want to match the shape but not lose any spatial info
        if stride != 1 or input_embed != output_embed:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_embed, output_embed, kernel_size= 1, stride = stride, bias= bias),
                nn.BatchNorm2d(output_embed)
            )

    def forward(self,x):
        # identity is x then follow the architecture: conv1 -> relu -> conv2 -> check for mis-match otherwise sum them up
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        # if there is a mis match
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.ReLU(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # This is the key difference! Final output channel = out_channels * 4
    # because you will first shrink it down channel wise not dimension, run a 3x3 kernel through it and upsized it back up by x4 to apply the fx + x

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()

        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv for spatial feature learning
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv to expand back to out_channels * 4
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # Downsample shortcut path if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Resnet(nn.Module):
    def __init__(self, block, block_counts,  num_classes, img_channels = 3):
        super().__init__()
        
        # We need to track the input channels to make sure they match each other to call for the downsample
        # Before the first layer, the input image is usually 224x224, it will go through a 7x7x64 and then some more downsample but the depth will start out at 64

        self.init_embed = 64
        self.block = block # we need to access the expansion later because
        # Start from the input, channels = 3 and we want output = 64, with kernel size 7, stride 2, padding 3 because start off at 224 and by doing this it will be 112x112 
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=7, stride=2, padding= 3, bias = False )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # now we maxpool it before putting it in the layers and blocks
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)

        # adding the layers
        # first layer stride = 1 because we just maxpool before this so the spatial size is good
        self.layer1 = self.layer_group(block, 64,block_counts[0],stride = 1)
        # after this the stride of the first one will be 2 since we are shrinking it down
        # only the first block will get stride 2 because we set stride = stride for it but after that it is hardcoded to stride = 1
        self.layer2 = self.layer_group(block, 128,block_counts[1],stride = 2)
        self.layer3 = self.layer_group(block, 256,block_counts[2],stride = 2)
        self.layer4 = self.layer_group(block, 512,block_counts[3],stride = 2)

        # after this we need to reduce the spatial size down to 1x1 so that we can convert spatial feature map into a vector for the fully connected classifier
        # use AdaptiveAvgPool2d because the size of the layer before it might not be exact 7x7 and this one is better and we are passing a tuple (1x1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        
        # fully connected layer
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def layer_group(self, block, out_channels, num_blocks, stride):
        layers = []
        # Since the first block of every layer will take the input from the above layers, we will need to use stride = stride to shrink it down
        layers.append(block(self.init_embed, out_channels, stride = stride ))
        # now the rest of the blocks in this layer use the same in and out 
        self.init_embed = out_channels * block.expansion
        # stride = 1 because we are not shrinking or anything here
        for _ in range (1, num_blocks):
            layers.append(block(self.init_embed,out_channels,stride = 1))

        # we are wrapping this list in a nn.Sequential so the model know to run this whole model
        # * is used to unpack the list basically calling out each blocks in that list 1 by 1 
        return nn.Sequential(*layers)
    
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
        # flatten x since x is still (B, 512,1,1) we need to flatten it out to use linear
        x = x.view(x.size(0),-1) # size(0) is the batch and view will reshape B and the rest of which is 512,1,1 into B and 512
        x = self.fc(x)

        return x

def ResNet18(img_channel=3, num_classes=1000):
    return Resnet(BasicBlock, [2, 2, 2, 2], num_classes, img_channel)

def ResNet36(img_channel=3, num_classes=1000):
    return Resnet(BasicBlock, [4, 4, 4, 4], num_classes, img_channel)

def ResNet50(img_channel=3, num_classes=1000):   
    return Resnet(Bottleneck, [3, 4, 6, 3], num_classes, img_channel)

def ResNet101(img_channel=3, num_classes=1000):
    return Resnet(Bottleneck, [3, 4, 23, 3], num_classes, img_channel)

def ResNet152(img_channel=3, num_classes=1000):
    return Resnet(Bottleneck, [3, 8, 36, 3], num_classes, img_channel)


def test_all_resnets():
    BATCH_SIZE = 4
    NUM_CLASSES = 1000
    IMG_CHANNELS = 3
    IMG_SIZE = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)

    models = {
        "ResNet18": ResNet18(num_classes=NUM_CLASSES, img_channel=IMG_CHANNELS),
        "ResNet36": ResNet36(num_classes=NUM_CLASSES, img_channel=IMG_CHANNELS),
        "ResNet50": ResNet50(num_classes=NUM_CLASSES, img_channel=IMG_CHANNELS),
        "ResNet101": ResNet101(num_classes=NUM_CLASSES, img_channel=IMG_CHANNELS),
        "ResNet152": ResNet152(num_classes=NUM_CLASSES, img_channel=IMG_CHANNELS),
    }

    for name, model in models.items():
        model = model.to(device)
        output = model(input_tensor)
        assert output.shape == (BATCH_SIZE, NUM_CLASSES), f"{name} output shape mismatch"
        print(f"{name} output: {output.shape}")


if __name__ == "__main__":
    test_all_resnets()
