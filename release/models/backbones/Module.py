import torch.nn as nn
import torch


class HBP(nn.Module):
    def __init__(self, inchannels, tempchannels, outchannels, kernel_size):
        '''
        @inchannels: 输入通道数
        @tempchannels： 中间通道数
        @outchannels
        @kernel_size: 平均池化的核
        '''
        super(HBP, self).__init__()

        self.proj0 = nn.Conv2d(inchannels, tempchannels, kernel_size=1, stride=1)
        self.proj1 = nn.Conv2d(inchannels, tempchannels, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(inchannels, tempchannels, kernel_size=1, stride=1)
        
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size)

        # fc layer
        self.fc_concat = torch.nn.Linear(tempchannels * 3, outchannels)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()



    def forward(self, x):
        x1, x2, x3 = x
        batch_size = x1.size(0)

        feature4_0 = self.proj0(x1)
        feature4_1 = self.proj1(x2)
        feature4_2 = self.proj2(x3)

        inter1 = feature4_0 * feature4_1
        inter2 = feature4_0 * feature4_2
        inter3 = feature4_1 * feature4_2

        inter1 = self.avgpool(inter1).view(batch_size, -1)
        inter2 = self.avgpool(inter2).view(batch_size, -1)
        inter3 = self.avgpool(inter3).view(batch_size, -1)


        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))


        result = torch.cat((result1, result2, result3), 1)
        result = self.fc_concat(result)
        return result

