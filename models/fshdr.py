import torch
import torch.nn as nn


class SDC(nn.Module):
    def __init__(self, nChannels, nFeatures):
        super(SDC, self).__init__()
        self.nChannels = nChannels
        self.nFeatures = nFeatures
        self.conv1 = nn.Conv2d(self.nChannels, self.nFeatures, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.conv2 = nn.Conv2d(self.nChannels, self.nFeatures, kernel_size=3, stride=1, padding=2, bias=True,
                               dilation=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.conv3 = nn.Conv2d(self.nChannels, self.nFeatures, kernel_size=3, stride=1, padding=3, bias=True,
                               dilation=3)
        self.relu3 = nn.LeakyReLU(negative_slope=0.3, inplace=False)

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(x))
        out3 = self.relu3(self.conv3(x))
        out = torch.cat((out1, out2, out3), 1)
        return out


class FSHDR(nn.Module):
    def __init__(self, nChannel=6):
        super(FSHDR, self).__init__()
        self.nFeat = 64

        self.conv1 = nn.Conv2d(nChannel, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(nChannel, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv2d(nChannel, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv4 = nn.Conv2d(2 * self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv5 = nn.Conv2d(2 * self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv6 = nn.Conv2d(self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu6 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv7 = nn.Conv2d(self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu7 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv8 = nn.Conv2d(self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu8 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv9 = nn.Conv2d(3 * self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu9 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv10 = nn.Conv2d(3 * self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu10 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv11 = nn.Conv2d(self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu11 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv12 = nn.Conv2d(self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu12 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv13 = nn.Conv2d(self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu13 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv14 = nn.Conv2d(3 * self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu14 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv15 = nn.Conv2d(3 * self.nFeat, self.nFeat, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu15 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sdc1 = SDC(3 * self.nFeat, self.nFeat)
        self.sdc2 = SDC(6 * self.nFeat, self.nFeat)

        self.conv16 = nn.Conv2d(9 * self.nFeat, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, inp1, inp2, inp3):

        x1 = self.relu1(self.conv1(inp1))
        x2 = self.relu2(self.conv2(inp2))
        x3 = self.relu3(self.conv3(inp3))

        s1 = torch.cat((x1, x2), 1)
        s2 = torch.cat((x2, x3), 1)
        s1 = self.relu4(self.conv4(s1))
        s2 = self.relu5(self.conv5(s2))

        x1 = self.relu6(self.conv6(x1))
        x2 = self.relu7(self.conv7(x2))
        x3 = self.relu8(self.conv8(x3))

        s1 = torch.cat((x1, x2, s1), 1)
        s2 = torch.cat((x2, x3, s2), 1)
        s1 = self.relu9(self.conv9(s1))
        s2 = self.relu10(self.conv10(s2))

        x1 = self.relu11(self.conv11(x1))
        x2 = self.relu12(self.conv12(x2))
        x3 = self.relu13(self.conv13(x3))

        s1 = torch.cat((x1, x2, s1), 1)
        s2 = torch.cat((x2, x3, s2), 1)
        s1 = self.relu14(self.conv14(s1))
        s2 = self.relu15(self.conv15(s2))

        x = torch.cat((x2, s1, s2), 1)
        x1 = self.sdc1(x)
        x2 = torch.cat((x, x1), 1)
        x2 = self.sdc2(x2)
        x3 = torch.cat((x, x1, x2), 1)

        x = self.conv16(x3)
        output = torch.sigmoid(x)

        return output


