import torch.nn as nn
import torch

from torch.nn import Parameter

class HBP(nn.Module):
    '''
    这里用的是平均池化核，而不是原caffe版本中的sum核
    l2normalize和normalize有什么区别(BN和norm不一样，所以应该是可以的)
    '''
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


class BasicBlock(nn.Module):
    def __init__(self, d, N, active_mode=0):
        '''
        active_mode：0：before 1: after, 2:None
        '''
        super(BasicBlock, self).__init__()
        self.W = Parameter(torch.Tensor(d, N))
        self.active_mode = active_mode
        if self.active_mode==0:
            self.activ = nn.ReLU()
    def forward(self, x):
        # x = x*(self.W.T)
        x = torch.mm(x, self.W.T)
        if self.active_mode==0:
            x = self.activ(x)
        return x
        

        

class MLB(nn.Module):
    def __init__(self, d, c, active_mode=0):
        '''
        @d: d<=min(x.view(batch,-1).size(0), y.view(batch, -1))
        @c: c=output
        # @kernel_size: 平均池化的核
        @active_mode: 0：before 1: after, 2:None
        '''
        super(MLB, self).__init__()
        self.active_mode = active_mode
        self.U = BasicBlock(d,256, active_mode=active_mode)
        self.V = BasicBlock(d,256, active_mode=active_mode)
        if self.active_mode==1:
            self.relu = nn.ReLU()
        self.P = torch.nn.Linear(d, c)
    def forward(self, x, y):
        '''
        @x,y : Batch*N
        '''
        # print(x.size())
        self.ux = self.U(x) # Batch*d
        self.vy = self.V(y)
        if self.active_mode==1:
            self.ux = self.relu(self.ux)
            self.vy = self.relu(self.vy)
        f = self.P(self.ux*self.vy)
        return f