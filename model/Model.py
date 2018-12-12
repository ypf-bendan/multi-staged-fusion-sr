import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self, channel_in):
        super(Conv_ReLU_Block, self).__init__()
        #res layer
        self.conv_0 = nn.Conv2d(in_channels=channel_in, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        #dense layer
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)

        #select layer
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        #active
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out_0 = self.relu(self.conv_0(residual))

        out_1 = self.relu(self.conv_1(x))

        out_2 = self.relu(self.conv_2(out_1))
        cat_1 = torch.cat((out_1, out_2), 1)

        out_3 = self.relu(self.conv_3(cat_1))
        cat_2 = torch.cat((cat_1, out_3), 1)

        out_4 = self.relu(self.conv_4(cat_2))
        cat_3 = torch.cat((cat_2, out_4), 1)
    
        out = torch.add(out_0, cat_3)
        out = self.relu(self.conv_5(out))

        return out
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.crb_1 = self.make_layer(Conv_ReLU_Block, 64)
        self.crb_2 = self.make_layer(Conv_ReLU_Block, 64)
        self.crb_3 = self.make_layer(Conv_ReLU_Block, 128)
        self.crb_4 = self.make_layer(Conv_ReLU_Block, 192)
        self.crb_5 = self.make_layer(Conv_ReLU_Block, 256)
        self.crb_6 = self.make_layer(Conv_ReLU_Block, 256)
        self.crb_7 = self.make_layer(Conv_ReLU_Block, 256)
        self.crb_8 = self.make_layer(Conv_ReLU_Block, 256)
        self.crb_9 = self.make_layer(Conv_ReLU_Block, 256)
        self.crb_10 = self.make_layer(Conv_ReLU_Block, 256)

        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.merge = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.res_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.res_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.res_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.res_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x    
        conv = self.relu(self.input(x))

        db_1  = self.crb_1(conv)
        db_2  = self.crb_2(db_1)
        cat_1 = torch.cat((db_1, db_2), 1)

        db_3  = self.crb_3(cat_1)
        cat_2 = torch.cat((db_1, db_2, db_3), 1)

        db_4  = self.crb_4(cat_2)
        cat_3 = torch.cat((db_1, db_2, db_3, db_4), 1)

        db_5  = self.crb_5(cat_3)
        cat_4 = torch.cat((db_2, db_3, db_4, db_5), 1)

        db_6  = self.crb_6(cat_4)
        cat_5 = torch.cat((db_3, db_4, db_5, db_6), 1)

        db_7  = self.crb_7(cat_5)
        cat_6 = torch.cat((db_4, db_5, db_6, db_7), 1)

        db_8  = self.crb_8(cat_6)
        cat_7 = torch.cat((db_5, db_6, db_7, db_8), 1)

        db_9  = self.crb_9(cat_7)
        cat_8 = torch.cat((db_6, db_7, db_8, db_9), 1)

        db_10  = self.crb_10(cat_8)
        cat_9 = torch.cat((db_7, db_8, db_9, db_10), 1)
        
        HR_com = self.relu(self.merge(cat_9))

        lc_1  = self.relu(self.res_1(conv)) 
        lc_2  = self.relu(self.res_2(lc_1))
        lc_3  = self.relu(self.res_3(lc_2))
        lc_4  = self.relu(self.res_4(lc_3))

        add_cat_lc = torch.add(HR_com, lc_4)
     
        out_1 = self.output(add_cat_lc)
        out = torch.add(out_1, residual)
        return out