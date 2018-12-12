import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self, in_channels):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input  = nn.Conv2d(in_channels=1,    out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1  = nn.Conv2d(in_channels=64,   out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9  = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv11 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv13 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv14 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv15 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv16 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv17 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv18 = nn.Conv2d(in_channels=128,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128,  out_channels=1,  kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, in_channels, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        conv0  = self.relu(self.input(x))
        conv1  = self.relu(self.conv1(conv0))
        conv2  = self.relu(self.conv2(torch.cat((conv0,  conv1), 1)))
        conv3  = self.relu(self.conv3(torch.cat((conv1,  conv2), 1)))
        conv4  = self.relu(self.conv4(torch.cat((conv2,  conv3), 1)))
        conv5  = self.relu(self.conv5(torch.cat((conv3,  conv4), 1)))
        conv6  = self.relu(self.conv6(torch.cat((conv4,  conv5), 1)))
        conv7  = self.relu(self.conv7(torch.cat((conv5,  conv6), 1)))
        conv8  = self.relu(self.conv8(torch.cat((conv6,  conv7), 1)))
        conv9  = self.relu(self.conv9(torch.cat((conv7,  conv8), 1)))
        conv10 = self.relu(self.conv10(torch.cat((conv8,  conv9), 1)))
        conv11 = self.relu(self.conv11(torch.cat((conv9,  conv10), 1)))
        conv12 = self.relu(self.conv12(torch.cat((conv10, conv11), 1)))
        conv13 = self.relu(self.conv13(torch.cat((conv11, conv12), 1)))
        conv14 = self.relu(self.conv14(torch.cat((conv12, conv13), 1)))
        conv15 = self.relu(self.conv15(torch.cat((conv13, conv14), 1)))
        conv16 = self.relu(self.conv16(torch.cat((conv14, conv15), 1)))
        conv17 = self.relu(self.conv17(torch.cat((conv15, conv16), 1)))
        conv18 = self.relu(self.conv18(torch.cat((conv16, conv17), 1)))

        out = self.output(torch.cat((conv17, conv18), 1))
        out = torch.add(out,residual)
        return out
 