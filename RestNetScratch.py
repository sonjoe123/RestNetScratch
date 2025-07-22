import  torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self,input_embed, output_embed, stride = 1, padding = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_embed,output_embed,kernel_size=3 ,stride= stride, padding = padding),
            nn.BatchNorm2d(output_embed),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_embed, output_embed,kernel_size=3, stride= stride, padding= padding),
            nn.BatchNorm2d(output_embed),
        )
        self.ReLU = nn.ReLU()


