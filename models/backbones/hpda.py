import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout, MaxPool2d, Sequential, Module
import torch

# Support: ['ResNet_50', 'ResNet_101', 'ResNet_152']


def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""

    return Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""

    return Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)



class LANet(Module):
    def __init__(self, in_channels, down_rate):
        super(LANet, self).__init__()
        self.conv1 = conv1x1(in_channels, in_channels//down_rate)
        self.relu = ReLU(inplace = True)
        self.conv2 = conv1x1(in_channels//down_rate, 1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return out


class PDNet_branch(Module):
    def __init__(self, inchannels, scale_index, branch=3):
        super(PDNet_branch, self).__init__()
        self.inchannels = inchannels
        self.branch = branch
        self.scale_index = scale_index
        
        if self.scale_index==0:
            self.downsample = None
            self.upsample_list = None
        else:
        # self.downsample = nn.AvgPool2d(kernel_size=2**scale_index, stride=2**scale_index)
            self.downsample = nn.AvgPool2d(kernel_size=2**scale_index)
            self.upsample_list = nn.ModuleList([nn.Upsample(scale_factor=2**scale_index, mode="bilinear") for i in range(branch)]) # TODO 需要改，使用方式不确定

        self.pd_branch_list = nn.ModuleList([LANet(self.inchannels, 2) for i in range(branch)])
    
    def forward(self, x):
        '''
        @mask_list: BNWH (BR*BA*W*H)
        @mask_up_list: NBWH
        '''
        if self.scale_index==0:
            out = x
        else:
            out = self.downsample(x)

        # print("downsampleing ", out.size())
        mask_list = [] # TODO 需要加速?
        mask_up_list = []
        for i in range(self.branch):
            mask = self.pd_branch_list[i](out) # N*1*W'*H'
            # print("get mask ", mask.size())
            if self.scale_index==0:
                mask_up = mask
            else:
                mask_up = self.upsample_list[i](mask)
                # mask_up = torch.sigmoid(self.upsample_list[i](mask))
            # print("mask upsamling", mask_up.size())
            mask_list.append(mask)
            mask_up_list.append(mask_up)
        # return torch.stack(mask_list, dim=1), torch.stack(mask_up_list, dim=1) # TODO 期望是BA*BR*1*W*H
        return torch.stack(mask_list, dim=0), torch.stack(mask_up_list, dim=0) # TODO 期望是BR*BA*1*W*H # 必须


class PDANet(Module):
    def __init__(self, inchannels, scale=3, branch=3):
        super(PDANet, self).__init__()
        self.inchannels = inchannels
        self.scale = scale
        self.branch = branch
        self.pd_scale_list = self._make_layer(inchannels, scale, branch)
        self.conv1_list = nn.ModuleList([conv1x1(inchannels*branch, inchannels) for i in range(scale)])
    
    def _make_layer(self, inchannels, scale, branch):
        net_list = []
        for i in range(scale):
            branch_net = PDNet_branch(inchannels, i, branch)
            net_list.append(branch_net)
        return nn.ModuleList(net_list)
    
    def forward(self, x):
        '''
        @mask_up_list: N*(S*B)*W*H
        @out: (BA*(S*C)*W*H)
        '''
        # mask_list = []
        mask_up_list = []
        scale_out_list = []
        for i in range(self.scale):
            mask_branch_list, mask_up_branch_list = self.pd_scale_list[i](x)
            # mask_list.append(mask_branch_list)
            mask_up_list.append(mask_up_branch_list) # [(B*N*1*W'*H'),...,] len()==3 
            attens_out_list = []
            for j in range(self.branch):
                attens_out = mask_up_branch_list[j]*x # N*1*W*H * NCWH #
                attens_out_list.append(attens_out)
            # scale attention out concat (N B*C W H)
            scale_attens_out = torch.cat(attens_out_list, 1)
            scale_out = self.conv1_list[i](scale_attens_out)

            scale_out_list.append(scale_out)
        # mask_list = torch.cat([o.view(o.size(0), o.size(1), o.size(2), -1) for o in mask_list], 1)
        # mask_up_list = torch.cat(mask_up_list, dim=0) # (S*B)*N*1*W'*H'
        mask_up_list = torch.stack(mask_up_list, dim=0) # S*B*N*1*W'*H'
        # mask_up_list = mask_up_list.transpose(0,1)
        # mask_up_list = mask_up_list.permute([1,0,2,3]) # # ==>N*(S*B)*1*W'*H'
        out = torch.cat(scale_out_list, dim=1) # (BA*(S*C)*W*H)
        # print("get pda out ", out.size())

        return mask_up_list, out


from .resnet import BasicBlock, Bottleneck
from .Module import HBP

class HPDA_res50(Module):
    def __init__(self, scale=1, branch=1, backbone="resnet50", block=Bottleneck,
                 global_branch=True, hbp=False, input_size=[112,112]):
        super(HPDA_res50, self).__init__()
        self.inplanes = 64
        self.hbp = hbp
        layers = [3, 4, 6, 3]
        self.global_branch = global_branch
        self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # 56*56
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1) # 28*28
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2) # 14*14
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2) # 7*7

        self.stem = Sequential(self.conv1, self.bn1,self.relu, self.maxpool,
                                self.layer1, self.layer2, self.layer3)
        # print(self.inplanes)
        self.downsample_4 = Sequential(
                conv1x1(self.inplanes, 512 * block.expansion, 2),
                BatchNorm2d(512 * block.expansion),
            )
        self.conv4_1 = block(self.inplanes, 512, 2, self.downsample_4)
        self.inplanes = 512 * block.expansion
        self.conv4_2 = block(self.inplanes, 512)
        self.conv4_3 = block(self.inplanes, 512)
        
        self.pda1 = PDANet(self.inplanes, scale=scale, branch=branch)
        self.pda2 = PDANet(self.inplanes, scale=scale, branch=branch)
        self.pda3 = PDANet(self.inplanes, scale=scale, branch=branch)

        self.scale = scale
        self.branch = branch

        if self.hbp:
            self.fc_local = HBP(self.inplanes*self.scale, self.inplanes*4, 512, 4)
        else:
            self.fc_local = Linear(self.inplanes*4*4*self.scale*3, 512) # (BA*(S*C*D)*W*H)
        self.fc_final = Linear(512, 512)
        
        if self.global_branch:
            # self.gap = nn.MaxPool2d(kernel_size=2, stride=1)
            # self.fc_global = Linear(self.inplanes*3*3, 512)
            self.gap = nn.MaxPool2d(kernel_size=4, stride=1)
            self.fc_global = Linear(self.inplanes, 512)
        
            self.fc_final = Linear(512*2, 512)
        
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
    
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)
    
    def forward(self, x):
        mask_list = list() # N*(S*B*D)*W*H
        pda_out_list = list() # list D*[(BA*(S*C)*W*H)]
        
        x = self.stem(x)
        # print("before conv4_1", x.size())
        x = self.conv4_1(x)
        # print("after conv4_1", x.size())

        mask_list_pda1, out_pda1 = self.pda1(x)
        pda_out_list.append(out_pda1)
        mask_list.append(mask_list_pda1)

        x = self.conv4_2(x)
        mask_list_pda2, out_pda2 = self.pda2(x)
        pda_out_list.append(out_pda2)
        mask_list.append(mask_list_pda2)

        x = self.conv4_3(x)
        mask_list_pda3, out_pda3 = self.pda3(x)
        pda_out_list.append(out_pda3)
        mask_list.append(mask_list_pda3)

        mask_list = torch.stack(mask_list, dim=0) # D*S*B*N*WH

        if self.global_branch:
            x = self.gap(x)
            # print("get gap result ", x.size())
            x = x.view(x.size(0), -1)
            global_feature = self.fc_global(x)
            
            if self.hbp:
                local_feature = self.fc_local(pda_out_list)
            else:
                local_out = torch.cat(pda_out_list, 1) # (BA*(S*C*D)*W*H)
                # print(local_out.size())
                local_feature = self.fc_local(local_out.view(local_out.size(0), -1))
            
            feature = torch.cat([local_feature, global_feature], 1)
            feature = self.fc_final(feature)
        else:
            if self.hbp:
                local_feature = self.fc_local(pda_out_list)
            else:
                local_out = torch.cat(pda_out_list, 1)
                local_feature = self.fc_local(local_out.view(local_out.size(0), -1))
            feature = self.fc_final(feature)
        return mask_list, feature

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)
