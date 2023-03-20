import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout, MaxPool2d, Sequential, Module
from torch.nn import LayerNorm
import torch
# import torch.functional as functional
# Support: ['ResNet_50', 'ResNet_101', 'ResNet_152']
from torch.nn.modules.dropout import AlphaDropout, Dropout2d


def conv3x3(in_planes, out_planes, stride = 3):
    """3x3 convolution with padding"""

    return Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""

    return Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        # self.bn1 = LayerNorm([2,3])
        self.relu = ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        # self.bn2 = LayerNorm([2,3])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        # self.bn1 = LayerNorm([2,3])
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        # self.bn2 = LayerNorm([2,3])
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Module):

    def __init__(self, input_size, block, layers, zero_init_residual=True, fc_num=1):
        super(ResNet, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        
        self.bn_o1 = BatchNorm2d(512 * block.expansion)
        self.dropout = Dropout(p=0)
        if input_size[0] == 112:
            if fc_num==2:
                self.fc = Linear(512 * block.expansion * 4 * 4, 256)
                self.fc_two = Linear(512 * block.expansion * 4 * 4, 256)
            else:
                self.fc = Linear(512 * block.expansion * 4 * 4, 512)
                self.fc_two = None
        else:
            if fc_num==2:
                self.fc = Linear(512 * block.expansion * 8 * 8, 256)
                self.fc_two = Linear(512 * block.expansion * 8 * 8, 256)
            else:
                self.fc = Linear(512 * block.expansion * 8 * 8, 512)
                self.fc_two = None
        # self.bn_o2 = BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),# add channel
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.bn_o1(x)
        x = self.dropout(x)
        fc_in = x.view(x.size(0), -1)
        x = self.fc(fc_in)
        # x = self.bn_o2(x)
        if self.fc_two:
            x2 = self.fc_two(fc_in)
            return x, x2
        return x
        # return torch.nn.functional.normalize(x)
    
    def forward_split(self, x, split_index=0, pad_direction="up"):
        if pad_direction=="up":
            padd = nn.ZeroPad2d(padding=(0,0,split_index,0))
            input = padd(x[:,:,split_index:,:])
        elif pad_direction=="down":
            B,C,H,W = x.shape
            padd = nn.ZeroPad2d(padding=(0,0,0,H-split_index))
            input = padd(x[:,:,:split_index,:])
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_o1(x)
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.bn_o2(x)

        return x


    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)    


class ResNet_split(Module):

    def __init__(self, input_size, block, layers, split_rate=56.7366/112, zero_init_residual = True):
        super(ResNet_split, self).__init__()
        '''
        split_rate*31==15: 特征尺寸上刚好分为一半，所以实现过程中没有四舍五入
        ...=14 前者小1，后者大1
        '''
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64
        self.split_rate = split_rate
        self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)# 56 # RF:7
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1) # 28 13-15 # RF:11
        
        self.layer1 = self._make_layer(block, 64, layers[0]) #
        # self.layer1_up = self._make_layer(block, 64, layers[0])
        self.layer2_up = self._make_layer(block, 128, layers[1], stride = 2) #14 7-8
        self.layer3_up = self._make_layer(block, 256, layers[2], stride = 2) # 7 4-4
        self.layer4_up = self._make_layer(block, 512, layers[3], stride = 2) # 4 2-2
        
        self.inplanes = 64*(block.expansion**1)
        # self.layer1_down = self._make_layer(block, 64, layers[0])
        self.layer2_down = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3_down = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4_down = self._make_layer(block, 512, layers[3], stride = 2)
        
        self.bn_o1_up = BatchNorm2d(512 * block.expansion)
        self.bn_o1_down = BatchNorm2d(512 * block.expansion)
        # self.dropout_up= Dropout(p=0.5)
        self.dropout_up= Dropout2d(p=0.1)
        # self.dropout_down = Dropout(p=0.5)
        if input_size[0] == 112:
            self.fc_up = Linear(512 * block.expansion * 4 * int(4*self.split_rate), 256) # TODO 原本是512
            self.fc_down = Linear(512 * block.expansion * 4 * int(4*self.split_rate), 256)
        else:
            self.fc_up = Linear(512 * block.expansion * 8 * int(8*self.split_rate), 256)
            self.fc_down = Linear(512 * block.expansion * 8 * int(8*self.split_rate), 256)
        # self.bn_o2 = BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),# add channel
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        B,C,H,W = x.shape
        # x_up = self.layer1_up(x[:,:,:int(H*self.split_rate)-1, :])
        x_up = self.layer2_up(x[:,:,:int(H*self.split_rate)-1, :])
        # x_up = self.layer2_up(x_up)
        x_up = self.layer3_up(x_up)
        x_up = self.layer4_up(x_up)

        x_up = self.bn_o1_up(x_up)
        x_up = self.dropout_up(x_up.permute(2,3,0,1)).permute(2,3,0,1)
        x_up = x_up.view(x_up.size(0), -1)
        x_up = self.fc_up(x_up)
        # x = self.bn_o2(x)
        
        # x_down = self.layer1_down(x[:,:,:int(H*self.split_rate)+1, :])
        # x_down = self.layer2_down(x[:,:,:int(H*self.split_rate)+1, :])
        x_down = self.layer2_down(x[:,:,int(H*self.split_rate)-1:, :])
        # x_down = self.layer2_down(x_down)
        x_down = self.layer3_down(x_down)
        x_down = self.layer4_down(x_down)

        x_down = self.bn_o1_down(x_down)
        # x_down = self.dropout_down(x_down)
        x_down = x_down.view(x_down.size(0), -1)
        x_down = self.fc_down(x_down)

        x_down = torch.nn.functional.normalize(x_down)
        x_up = torch.nn.functional.normalize(x_up)

        return x_up, x_down 
        # return torch.nn.functional.normalize(x)
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)    


class SpaceSliceStrategy_subject(Module):
    def __init__(self, layers, inplanes, block=BasicBlock, zero_init_residual=True):
        super(SpaceSliceStrategy_subject, self).__init__()
        self.inplanes = inplanes

        self.num_module = len(layers)
        self.module_list = []

        if self.num_module>3:
            self.layer1 = self._make_layer(block, 64, layers[-4])
            self.module_list.append(self.layer1)
        
        if self.num_module>2:
            self.layer2 = self._make_layer(block, 128, layers[-3], stride = 2)
            self.module_list.append(self.layer2)
    
        if self.num_module>1:
            self.layer3 = self._make_layer(block, 256, layers[-2], stride = 2)
            self.module_list.append(self.layer3)
        
        self.layer4 = self._make_layer(block, 512, layers[-1], stride = 2)
        self.bn_o1 = BatchNorm2d(512 * block.expansion)
        self.module_list.append(self.layer4)
        self.module_list.append(self.bn_o1)

        self.subject = nn.Sequential(*self.module_list)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),# add channel
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        out = self.subject(x)
        return out


class ResNet_split_abla(Module):

    def __init__(self, input_size, block, layers, split_rate=56.7366/112, zero_init_residual = True, dist=False, SSS_location=1):
        super(ResNet_split_abla, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.inplanes = 64
        self.split_rate = split_rate
        self.module_list = [self.conv1, self.bn1, self.relu, self.maxpool]

        if SSS_location>1:
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.module_list.append(self.layer1)
        
        if SSS_location>2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
            self.module_list.append(self.layer2)
    
        if SSS_location>3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
            self.module_list.append(self.layer3)
        
        self.backbone = nn.Sequential(*self.module_list)
        
        self.SSS_up = SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block, inplanes=self.inplanes)
        self.SSS_down = SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block, inplanes=self.inplanes)
        
        # self.drop_out_up= Dropout2d(p=0.1)
        # self.dropout_down = Dropout(p=0.5)
        if input_size[0] == 112:
            self.fc_up = Linear(512 * block.expansion * 4 * 2, 256) # TODO 原本是512
            # self.fc_up = Linear(512 * block.expansion * 4 * 1, 256) # TODO 原本是512
            self.fc_down = Linear(512 * block.expansion * 4 * 2, 256)
            # self.fc_down = Linear(512 * block.expansion * 4 * 3, 256)
        else:
            self.fc_up = Linear(512 * block.expansion * 8 * int(8*self.split_rate), 256)
            self.fc_down = Linear(512 * block.expansion * 8 * int(8*self.split_rate), 256)

        # self.bn_o2 = BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),# add channel
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)
    
    def forward(self, x):
        out = self.backbone(x)
        B,C,H,W = out.shape  
        
        SSS_out_up = self.SSS_up(out[:,:,:int(H*self.split_rate)-1, :]) # int(H*self.split_rate)
        SSS_out_down = self.SSS_down(out[:,:,int(H*self.split_rate)-1:, :])

        # SSS_out_up = self.drop_out_up(SSS_out_up.permute(2,3,0,1)).permute(2,3,0,1)
        # SSS_out_up = self.drop_out_up(SSS_out_up)

        SSS_out_up = SSS_out_up.view(SSS_out_up.size(0), -1)
        SSS_out_down = SSS_out_down.view(SSS_out_down.size(0), -1)

        feature_up = self.fc_up(SSS_out_up)
        feature_down = self.fc_down(SSS_out_down)
        
        feature_up = torch.nn.functional.normalize(feature_up)
        feature_down = torch.nn.functional.normalize(feature_down)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature_up, feature_down
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, kernel_size=7, pool_size = (1, 2, 3), channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.avg_spp = SPPModule('avg', self.pool_size)
        self.max_spp = SPPModule('max', self.pool_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2,
                      dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in self.pool_size]))
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_input = self.avg_spp(x) + self.max_spp(x)
        channel_scale = self.channel(channel_input)

        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1) # torch.max 会返回最大值和索引，所以需要【0】，两个torch，的操作都会降低维度
        spatial_scale = self.spatial(spatial_input)

        x_age = (x * channel_scale + x * spatial_scale) * 0.5

        x_id = x - x_age

        return x_id, x_age


class ResNet_split_attention(ResNet_split):

    def __init__(self, input_size, block, layers, split_rate=56.7366/112, zero_init_residual = True, dist=False):
        super(ResNet_split_attention, self).__init__(input_size, block, layers)

        out_neurons = 2
        self.attention = AttentionModule(channels=128*block.expansion)
        self.mask_output_layer = nn.Sequential(
            nn.BatchNorm2d(128*block.expansion),
            nn.Flatten(),
            nn.Linear(128*block.expansion * 14 * int(14*split_rate), 512),
            nn.LeakyReLU(0.2, inplace=True) if dist else nn.ReLU(inplace=True),
            nn.Linear(512, out_neurons),
        )

        self.softmax = nn.Softmax(dim=1)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        B,C,H,W = x.shape
        # x_up = self.layer1_up(x[:,:,:int(H*self.split_rate)-1, :])
        x_up = self.layer2_up(x[:,:,:int(H*self.split_rate)-1, :])
        # print(x_up.shape)
        x_up, x_up_mask = self.attention(x_up)
        mask_out = self.mask_output_layer(x_up_mask)
        # x_up = self.layer2_up(x_up)
        x_up = self.layer3_up(x_up)
        x_up = self.layer4_up(x_up)

        x_up = self.bn_o1_up(x_up)
        x_up = self.dropout_up(x_up)
        x_up = x_up.view(x_up.size(0), -1)
        x_up = self.fc_up(x_up)
        # x = self.bn_o2(x)
        
        # x_down = self.layer1_down(x[:,:,:int(H*self.split_rate)+1, :])
        # x_down = self.layer2_down(x[:,:,:int(H*self.split_rate)+1, :])
        x_down = self.layer2_down(x[:,:,int(H*self.split_rate)-1:, :])
        # x_down = self.layer2_down(x_down)
        x_down = self.layer3_down(x_down)
        x_down = self.layer4_down(x_down)

        x_down = self.bn_o1_down(x_down)
        x_down = self.dropout_down(x_down)
        x_down = x_down.view(x_down.size(0), -1)
        x_down = self.fc_down(x_down)

        x_down = torch.nn.functional.normalize(x_down)
        x_up = torch.nn.functional.normalize(x_up)
        
        # no_mask_score = self.softmax(mask_out)[:,0].unsqueeze(1)
        # no_mask_score = torch.clamp(self.softmax(mask_out)[:,0].unsqueeze(1), min=0.3)
        # x_down = x_down * no_mask_score
        # no_mask_score = 1 - self.softmax(mask_out)[:,1].unsqueeze(1)    
        # x_down = x_down * no_mask_score * 0.1
        # mask_score = torch.max(self.softmax(mask_out), dim=1)[0].unsqueeze(1)
        # x_up = x_up * mask_score * 0.1

        return x_up, x_down, mask_out
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)     


class ResNet_split_attention_abla(ResNet_split):

    def __init__(self, input_size, block, layers, split_rate=56.7366/112, zero_init_residual = True, dist=False):
        super(ResNet_split_attention_abla, self).__init__(input_size, block, layers)

        out_neurons = 2
        self.attention = AttentionModule(channels=64*block.expansion, kernel_size=7)
        self.mask_output_layer = nn.Sequential(
            nn.BatchNorm2d(64*block.expansion),
            MaxPool2d(kernel_size = 3, stride = 2),
            nn.Flatten(),
            nn.Linear(64*block.expansion * 13 * 6, 512),
            nn.LeakyReLU(0.2, inplace=True) if dist else nn.ReLU(inplace=True),
            nn.Linear(512, out_neurons),
        )

        self.softmax = nn.Softmax(dim=1)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        B,C,H,W = x.shape
        # x_up = self.layer1_up(x[:,:,:int(H*self.split_rate)-1, :])
        x_up, x_up_mask = self.attention(x[:,:,:int(H*self.split_rate), :])
        mask_out = self.mask_output_layer(x_up_mask)

        x_up = self.layer2_up(x_up)
        # print(x_up.shape)
        
        # x_up = self.layer2_up(x_up)
        x_up = self.layer3_up(x_up)
        x_up = self.layer4_up(x_up)

        x_up = self.bn_o1_up(x_up)
        x_up = self.dropout_up(x_up)
        x_up = x_up.view(x_up.size(0), -1)
        x_up = self.fc_up(x_up)
        # x = self.bn_o2(x)
        
        # x_down = self.layer1_down(x[:,:,:int(H*self.split_rate)+1, :])
        # x_down = self.layer2_down(x[:,:,:int(H*self.split_rate)+1, :])
        x_down = self.layer2_down(x[:,:,int(H*self.split_rate):, :])
        # x_down = self.layer2_down(x_down)
        x_down = self.layer3_down(x_down)
        x_down = self.layer4_down(x_down)

        x_down = self.bn_o1_down(x_down)
        x_down = self.dropout_down(x_down)
        x_down = x_down.view(x_down.size(0), -1)
        x_down = self.fc_down(x_down)

        x_down = torch.nn.functional.normalize(x_down)
        x_up = torch.nn.functional.normalize(x_up)
        # no_mask_score = self.softmax(mask_out)[:,0].unsqueeze(1)
        # no_mask_score = torch.clamp(self.softmax(mask_out)[:,0].unsqueeze(1), min=0.3)
        no_mask_score = 1 - self.softmax(mask_out)[:,1].unsqueeze(1)    
        x_down = x_down * no_mask_score * 0.1
        mask_score = torch.max(self.softmax(mask_out), dim=1)[0].unsqueeze(1)
        x_up = x_up * mask_score * 0.1

        return x_up, x_down, mask_out
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f) 

class ResNet_split3(Module):

    def __init__(self, input_size, block, layers, split_rate=56.7366/112, zero_init_residual = True):
        super(ResNet_split3, self).__init__()
        '''
        split_rate*31==15: 特征尺寸上刚好分为一半，所以实现过程中没有四舍五入
        ...=14 前者小1，后者大1
        '''
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64
        self.split_rate = split_rate
        self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)# 59
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace = True)
        self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1) # 31 15-16
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer1_up = self._make_layer(block, 64, layers[0])
        # self.layer2_up = self._make_layer(block, 128, layers[1], stride = 2) #16 8-8
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2) #16 8-8
        self.layer3_up = self._make_layer(block, 256, layers[2], stride = 2) # 8
        self.layer4_up = self._make_layer(block, 512, layers[3], stride = 2) # 4
        
        self.inplanes = 64*(block.expansion**2)
        # self.layer1_down = self._make_layer(block, 64, layers[0])
        # self.layer2_down = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3_down = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4_down = self._make_layer(block, 512, layers[3], stride = 2)
        
        self.bn_o1_up = BatchNorm2d(512 * block.expansion)
        self.bn_o1_down = BatchNorm2d(512 * block.expansion)
        self.dropout_up= Dropout(p=0.5)
        self.dropout_down = Dropout(p=0.5)
        if input_size[0] == 112:
            self.fc_up = Linear(512 * block.expansion * 4 * int(4*self.split_rate), 256) # TODO 原本是512
            self.fc_down = Linear(512 * block.expansion * 4 * int(4*self.split_rate), 256)
        else:
            self.fc_up = Linear(512 * block.expansion * 8 * int(8*self.split_rate), 256)
            self.fc_down = Linear(512 * block.expansion * 8 * int(8*self.split_rate), 256)
        # self.bn_o2 = BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),# add channel
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        B,C,H,W = x.shape
        # x_up = self.layer1_up(x[:,:,:int(H*self.split_rate)-1, :])
        # x_up = self.layer2_up(x[:,:,:int(H*self.split_rate)-1, :])
        # x_up = self.layer2_up(x_up)
        x_up = self.layer3_up(x[:,:,:int(H*self.split_rate)-1, :])
        # x_up = self.layer3_up(x_up)
        x_up = self.layer4_up(x_up)

        x_up = self.bn_o1_up(x_up)
        x_up = self.dropout_up(x_up)
        x_up = x_up.view(x_up.size(0), -1)
        x_up = self.fc_up(x_up)
        # x = self.bn_o2(x)
        
        # x_down = self.layer1_down(x[:,:,:int(H*self.split_rate)+1, :])
        # x_down = self.layer2_down(x[:,:,:int(H*self.split_rate)+1, :])
        # x_down = self.layer2_down(x_down)
        x_down = self.layer3_down(x[:,:,:int(H*self.split_rate)+1, :])
        # x_down = self.layer3_down(x_down)
        x_down = self.layer4_down(x_down)

        x_down = self.bn_o1_down(x_down)
        x_down = self.dropout_down(x_down)
        x_down = x_down.view(x_down.size(0), -1)
        x_down = self.fc_down(x_down)

        return x_up, x_down 
        # return torch.nn.functional.normalize(x)
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)    


def ResNet_18(input_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, BasicBlock, [2, 2, 2, 2], **kwargs)

    return model

def ResNet_34(input_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, BasicBlock, [3, 4, 6, 3], **kwargs)

    return model

def ResNet_50(input_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ResNet_101(input_size, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def ResNet_152(input_size, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


def ResNet_50_split(input_size, **kwargs):
    """Constructs a ResNet_split-50 model.
    """
    model = ResNet_split(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model

def ResNet_50_split_abla(input_size, **kwargs):
    """Constructs a ResNet_split-50 model.
    """
    model = ResNet_split_abla(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model

def ResNet_101_split(input_size, **kwargs):
    """Constructs a ResNet_split-101 model.
    """
    model = ResNet_split(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def ResNet_152_split(input_size, **kwargs):
    """Constructs a ResNet_split-152 model.
    """
    model = ResNet_split(input_size, Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


def ResNet_50_split_attention(input_size, **kwargs):
    model = ResNet_split_attention(input_size, Bottleneck, [3, 4, 6, 3],**kwargs)
    return model


def ResNet_50_split_attention_abla(input_size, **kwargs):
    model = ResNet_split_attention_abla(input_size, Bottleneck, [3, 4, 6, 3],**kwargs)
    return model