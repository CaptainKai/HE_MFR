import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.dropout import AlphaDropout, Dropout2d


class ConvPrelu(nn.Module):

    def __init__(self, in_channels, out_channels, filter='gaussian', prelu_init=0.25, mean=0, std=0.01,**kwargs):
        super(ConvPrelu, self).__init__()
        if filter=="xavier":
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            torch.nn.init.xavier_uniform_(self.conv.weight)
            self.prelu = nn.PReLU(num_parameters=out_channels, init=prelu_init) # default=false, init=0.25
        elif filter=="gaussian":
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            torch.nn.init.normal_(self.conv.weight, mean=mean, std=std)# default=（0,0.01）
            self.prelu = nn.PReLU(num_parameters=out_channels, init=prelu_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class SeModule(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(SeModule, self).__init__()
        '''
        添加reduction，原代码为16
        '''
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=in_channels, out_features=round(in_channels / reduction))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=round(in_channels / reduction), out_features=in_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        original_out = x
        
        out = self.globalAvgPool(x)
        out = out.view(original_out.size(0), original_out.size(1))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(original_out.size(0), original_out.size(1),1,1)
        out = out * original_out

        return out

class SimpleResidualUnit(nn.Module):
    
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, use_se=False):
        super(SimpleResidualUnit, self).__init__()
        self.conv1 = ConvPrelu(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvPrelu(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.use_se = use_se
        if self.use_se:
            self.se = SeModule(in_channels)
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.use_se:
            out = self.se(out)

        out+=residual

        return out


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


class SimpleResidualBackbone(nn.Module):

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112]):
        super(SimpleResidualBackbone, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes

        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0])

        self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1])

        self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[2])

        self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer4 = self._make_layer(512, base_layer=block,layers_num=layers[3])

        self.fc5 = nn.Linear(512*7*7, 512)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)

    def _make_layer(self, in_channels, base_layer, layers_num):
        layers=[]
        for i in range(layers_num):
            layers.append(base_layer(in_channels, use_se=self.use_se))
            self.use_se = False# TODO
        return nn.Sequential(*layers)


    def forward(self, x):
        conv1 = self.conv1(x)
        layer1 = self.layer1(conv1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        conv3 = self.conv3(layer2)
        layer3 = self.layer3(conv3)
        conv4 = self.conv4(layer3)
        layer4 = self.layer4(conv4)
        layer4 = layer4.view(layer4.size(0), -1)

        feature = self.fc5(layer4)
        feature = torch.nn.functional.normalize(feature)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature

    def forward_split(self, x, split_index=0, pad_direction="up"):
        if pad_direction=="up":
            padd = nn.ZeroPad2d(padding=(0,0,split_index,0))
            input = padd(x[:,:,split_index:,:])
        elif pad_direction=="down":
            B,C,H,W = x.shape
            padd = nn.ZeroPad2d(padding=(0,0,0,H-split_index))
            input = padd(x[:,:,:split_index,:])
        
        conv1 = self.conv1(input)
        layer1 = self.layer1(conv1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        conv3 = self.conv3(layer2)
        layer3 = self.layer3(conv3)
        conv4 = self.conv4(layer3)
        layer4 = self.layer4(conv4)
        layer4 = layer4.view(layer4.size(0), -1)

        feature = self.fc5(layer4)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)
    
    def get_layer4(self, x):
        conv1 = self.conv1(x)
        layer1 = self.layer1(conv1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        conv3 = self.conv3(layer2)
        layer3 = self.layer3(conv3)
        conv4 = self.conv4(layer3)
        layer4 = self.layer4(conv4)
        layer4 = layer4.view(layer4.size(0), -1)

        return layer4


class SimpleResidual_split_Branch(nn.Module):

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112]):
        super(SimpleResidual_split_Branch, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes

        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0])

        self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1])

        self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[2])

        self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer4 = self._make_layer(512, base_layer=block,layers_num=layers[3])

        self.fc5 = nn.Linear(512*7*4, 256)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)

    def _make_layer(self, in_channels, base_layer, layers_num):
        layers=[]
        for i in range(layers_num):
            layers.append(base_layer(in_channels, use_se=self.use_se))
            self.use_se = False# TODO
        return nn.Sequential(*layers)


    def forward(self, x):
        conv1 = self.conv1(x)
        layer1 = self.layer1(conv1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        conv3 = self.conv3(layer2)
        layer3 = self.layer3(conv3)
        conv4 = self.conv4(layer3)
        layer4 = self.layer4(conv4)
        layer4 = layer4.view(layer4.size(0), -1)

        feature = self.fc5(layer4)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature

    def forward_split(self, x, split_index=0, pad_direction="up"):
        if pad_direction=="up":
            padd = nn.ZeroPad2d(padding=(0,0,split_index,0))
            input = padd(x[:,:,split_index:,:])
        elif pad_direction=="down":
            B,C,H,W = x.shape
            padd = nn.ZeroPad2d(padding=(0,0,0,H-split_index))
            input = padd(x[:,:,:split_index,:])
        
        conv1 = self.conv1(input)
        layer1 = self.layer1(conv1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        conv3 = self.conv3(layer2)
        layer3 = self.layer3(conv3)
        conv4 = self.conv4(layer3)
        layer4 = self.layer4(conv4)
        layer4 = layer4.view(layer4.size(0), -1)

        feature = self.fc5(layer4)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class SimpleResidual_fullsplit_Backbone(nn.Module):

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112):
        super(SimpleResidual_fullsplit_Backbone, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes
        self.split_rate = split_rate
        self.layer_up = SimpleResidual_split_Branch(layers=layers, block=block, input_size=input_size)
        self.layer_down = SimpleResidual_split_Branch(layers=layers, block=block, input_size=input_size)
    
    def forward(self, x):
        
        B,C,H,W = x.shape  
        
        feature_up = self.layer_up(x[:,:,:int(H*self.split_rate)-1, :])
        feature_down = self.layer_down(x[:,:,int(H*self.split_rate)-1:, :])
        
        feature_up = torch.nn.functional.normalize(feature_up)
        feature_down = torch.nn.functional.normalize(feature_down)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature_up, feature_down


    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class SpaceSliceStrategy_subject(nn.Module):
    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train'):
        super(SpaceSliceStrategy_subject, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes

        self.num_module = len(layers)
        self.module_list = []
        
        self.layer4 = self._make_layer(512, base_layer=block,layers_num=layers[-1])
        
        # mode 2
        self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.module_list.append(self.layer4)
        self.module_list.append(self.conv4)

        if self.num_module>1:
            self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[-2])
            self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier')
            self.module_list.append(self.layer3)
            self.module_list.append(self.conv3)

        if self.num_module>2:
            self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[-3])
            self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')
            self.module_list.append(self.layer2)
            self.module_list.append(self.conv2)

        if self.num_module>3:
            self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[-4])
            self.module_list.append(self.layer1)

        self.subject = nn.ModuleList(self.module_list)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)

    def _make_layer(self, in_channels, base_layer, layers_num):
        layers=[]
        for i in range(layers_num):
            layers.append(base_layer(in_channels, use_se=self.use_se))
            self.use_se = False# TODO
        return nn.Sequential(*layers)


    def forward(self, x):
        out = x
        for i in range(len(self.module_list)-1, -1, -1):
            out = self.subject[i](out)

        return out


class SimpleResidual_split_abla(nn.Module):
    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112, SSS_location=4):
        #sss_location: 1~4
        super(SimpleResidual_split_abla, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes
        self.split_rate = split_rate
        self.module_list = []
        
        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier') # 56 i:55/57 o:28/29
        self.module_list.append(self.conv1)

        if SSS_location>1:
            self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0])
            self.module_list.append(self.layer1)

        
        if SSS_location>2:
            self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')# 28 i:28/29 o:14/15
            self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1])
            self.module_list.append(self.conv2)
            self.module_list.append(self.layer2)

        if SSS_location>3:
            self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier') # 14 i:13 o:7 i:14/15 o:7/8
            self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[2])
            self.module_list.append(self.conv3)
            self.module_list.append(self.layer3)

        self.backbone = nn.Sequential(*self.module_list)
        
        self.SSS_up = SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block)
        self.SSS_down = SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block)

        # self.drop_out_up = Dropout2d(p=0.1)
        if SSS_location==4:
            self.fc_up = nn.Linear(512*7*3, 256) # last:3 # TODO
        else:
            self.fc_up = nn.Linear(512*7*4, 256)
        self.fc_down = nn.Linear(512*7*4, 256)# last:4

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)
    
    def _make_layer(self, in_channels, base_layer, layers_num):
        layers=[]
        for i in range(layers_num):
            layers.append(base_layer(in_channels, use_se=self.use_se))
            self.use_se = False# TODO
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        out = self.backbone(x)
        B,C,H,W = out.shape  
        
        SSS_out_up = self.SSS_up(out[:,:,:int(H*self.split_rate)-1, :]) # int(H*self.split_rate)-1
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


def SimpleResnet_20(**kwargs):
    """Constructs a SimpleResnet_20 model.
    """
    model = SimpleResidualBackbone([1, 2, 4, 1], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_36(**kwargs):
    """Constructs a SimpleResnet_36 model.
    """
    model = SimpleResidualBackbone([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_64(**kwargs):
    """Constructs a SimpleResnet_64 model.
    """
    model = SimpleResidualBackbone([3, 8, 16, 3], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_20_Se(**kwargs):
    """Constructs a SimpleResnet_20 model.
    """
    model = SimpleResidualBackbone([1, 2, 4, 1], use_se=True, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_36_Se(**kwargs):
    """Constructs a SimpleResnet_36 model.
    """
    model = SimpleResidualBackbone([2, 4, 8, 2], use_se=True, block=SimpleResidualUnit,  **kwargs)
    return model


def SimpleResnet_64_Se(**kwargs):
    """Constructs a SimpleResnet_64 model.
    """
    model = SimpleResidualBackbone([3, 8, 16, 3], use_se=True, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_fullsplit_36(**kwargs):
    """Constructs a SimpleResnet_fullsplit_36 model.
    """
    model = SimpleResidual_fullsplit_Backbone([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_split_abla_36(**kwargs):
    """Constructs a SimpleResnet_fullsplit_36 model.
    """
    model = SimpleResidual_split_abla([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model