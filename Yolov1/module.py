import torch
import torch.nn as nn
import hyperparameter as hp

class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        #p1
        self.conv1 = nn.Conv2d(3, 64, (7, 7), padding=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    #(batch size, 64, h/4, w/4)

        #p2
        self.conv2 = nn.Conv2d(64, 192, (3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    #(batch size, 192, h/8, w/8)

        #p3
        self.conv3 = nn.Conv2d(192, 128, (1, 1), stride=1)
        self.conv4 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, (1, 1), stride=1)
        self.conv6 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   #(batchsize, 512, h/16, w/16)

        #p4
        self.conv7 = nn.Conv2d(512, 256, (1, 1))
        self.conv8 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 256, (1, 1))
        self.conv10 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(512, 256, (1, 1))
        self.conv12 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv13 = nn.Conv2d(512, 256, (1, 1))
        self.conv14 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv15 = nn.Conv2d(512, 512, (1, 1))
        self.conv16 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)      #(batchsize, 1024, h/32, w/32)

        #p5
        self.conv17 = nn.Conv2d(1024, 512, (1, 1))
        self.conv18 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv19 = nn.Conv2d(1024, 512, (1, 1))
        self.conv20 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.conv21 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv22 = nn.Conv2d(1024, 1024, (3, 3), stride=2, padding=1)    #(batchsize, 1024, h/64, w/64)

        #p6
        self.conv23 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.conv24 = nn.Conv2d(1024, 1024, (3, 3), padding=1)      #(batchsize, 1024, h/64, w/64)

        #p7
        self.fc1 = nn.Linear(1024*hp.height//64*hp.width//64, 4096)   #(4096)

        #p8
        self.fc2 = nn.Linear(4096, 25*7*7)  #(7 ,7 ,25)

        #leakyrelu
        self.LeakR = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, input):

        #l1

        output = self.LeakR(self.conv1(input))
        output = self.pool1(output)

        #l2

        output = self.LeakR(self.conv2(output))
        output = self.pool2(output)

        #l3

        output = self.LeakR(self.conv3(output))
        output = self.LeakR(self.conv4(output))
        output = self.LeakR(self.conv5(output))
        output = self.LeakR(self.conv6(output))
        output = self.pool3(output)

        #l4

        output = self.LeakR(self.conv7(output))
        output = self.LeakR(self.conv8(output))
        output = self.LeakR(self.conv9(output))
        output = self.LeakR(self.conv10(output))
        output = self.LeakR(self.conv11(output))
        output = self.LeakR(self.conv12(output))
        output = self.LeakR(self.conv13(output))
        output = self.LeakR(self.conv14(output))
        output = self.LeakR(self.conv15(output))
        output = self.LeakR(self.conv16(output))
        output = self.pool4(output)

        #l5

        output = self.LeakR(self.conv17(output))
        output = self.LeakR(self.conv18(output))
        output = self.LeakR(self.conv19(output))
        output = self.LeakR(self.conv20(output))
        output = self.LeakR(self.conv21(output))
        output = self.LeakR(self.conv22(output))

        #l6

        output = self.LeakR(self.conv23(output))
        output = self.LeakR(self.conv24(output))

        #l7

        output = self.dropout(self.fc1(output.view(output.size(0), -1)))    #(batchsize, 1024*h/64*w/64)

        #l8

        output = self.fc2(output)       #(batchsize, 7*7*30)
        output = output.view(-1, 7, 7, 25)
        return output

