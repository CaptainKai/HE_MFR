import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, LayerNorm
from torch.nn.modules.dropout import AlphaDropout, Dropout2d

from .cbam import CBAM
from .bam import BAM
from .xcos import PCM_AM

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
    
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, use_se=False, use_cbam=False):
        super(SimpleResidualUnit, self).__init__()
        self.conv1 = ConvPrelu(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvPrelu(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.use_se = use_se
        if self.use_se:
            self.se = SeModule(in_channels)
        if use_cbam:
            self.cbam = CBAM( in_channels, 16 )
        else:
            self.cbam = None
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.use_se:
            out = self.se(out)
        if self.cbam:
            out = self.cbam(out)

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
            # nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in self.pool_size]))
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
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

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], att_type=None, ln=False, fc=True, fc_num=1, norm=True, plugin=False):
        super(SimpleResidualBackbone, self).__init__()
        self.phase = phase
        self.use_se = use_se
        self.fc = fc
        self.norm = norm
        # self.num_classes = num_classes

        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier')

        if att_type=='BAM':
            self.bam1 = BAM(64)
            self.bam2 = BAM(128)
            self.bam3 = BAM(256)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0], att_type=att_type)

        self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1], att_type=att_type)

        self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[2], att_type=att_type)
        if ln:
            self.ln = LayerNorm([256, 14,14])
        else:
            self.ln=None
        
        self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer4 = self._make_layer(512, base_layer=block,layers_num=layers[3], att_type=att_type)
        
        if plugin=="PCM_AM":
            self.plugin = PCM_AM()
        else:
            self.plugin = None
        
        if self.fc:
            if fc_num==2:
                self.fc5 = nn.Linear(512*7*7, 256)
                self.fc5_two = nn.Linear(512*7*7, 256)
            else:
                self.fc5 = nn.Linear(512*7*7, 512)
                self.fc5_two = None
        else:
            if fc_num==2:
                self.fc5 = nn.Conv2d(512, 32, bias=False, kernel_size=1)
                self.fc5_two = nn.Conv2d(512, 32, bias=False, kernel_size=1)
                torch.nn.init.xavier_uniform_(self.fc5.weight)
                torch.nn.init.xavier_uniform_(self.fc5_two.weight)
            else: # ! 无用 -- 论文里好像是共享的 --
                self.fc5 = nn.Conv2d(512, 32, bias=False, kernel_size=1)
                torch.nn.init.xavier_uniform_(self.fc5.weight)
                self.fc5_two = None

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)

    def _make_layer(self, in_channels, base_layer, layers_num, att_type):
        layers=[]
        for i in range(layers_num):
            layers.append(base_layer(in_channels, use_se=self.use_se, use_cbam=att_type=='CBAM'))
            self.use_se = False# TODO
        return nn.Sequential(*layers)


    def forward(self, x, paral=False):
        B = x.size(0)
        conv1 = self.conv1(x)
        layer1 = self.layer1(conv1)
        if not self.bam1 is None:
            layer1 = self.bam1(layer1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        if not self.bam2 is None:
            layer2 = self.bam2(layer2)
        conv3 = self.conv3(layer2)
        layer3 = self.layer3(conv3)
        if not self.bam3 is None:
            layer3 = self.bam3(layer3)
        if self.ln:
            layer3 = self.ln(layer3)
        
        conv4 = self.conv4(layer3)
        layer4 = self.layer4(conv4)
        if self.fc:
            layer4 = layer4.view(layer4.size(0), -1)

        if paral:
            feature = self.fc5(layer4[:B//2])
            if self.fc5_two is None:
                self.fc5_two = self.fc5
            feature2 = self.fc5_two(layer4[B//2:])
            if self.norm:
                feature = torch.nn.functional.normalize(feature, dim=1)
                feature2 = torch.nn.functional.normalize(feature2, dim=1)
            if self.plugin is not None:
                s = self.plugin(feature, feature2)
                # feature = torch.flatten(feature, start_dim=1) # 尝试一下是否可以解决问题 # !
                # feature2 = torch.flatten(feature2, start_dim=1)
                return feature, feature2, s
            return feature, feature2
        else:    
            feature = self.fc5(layer4)
            if self.norm:
                feature = torch.nn.functional.normalize(feature, dim=1) # dim默认就是1
            elif self.phase == 'test':
                feature = torch.flatten(feature, start_dim=1) # 尝试一下是否可以解决问题 # !
            if self.fc5_two:
                feature2 = self.fc5_two(layer4)
                if self.norm:
                    feature2 = torch.nn.functional.normalize(feature2,dim=1)
                elif self.phase == 'test':
                    feature2 = torch.flatten(feature2, start_dim=1)    
            # if self.plugin is not None: #无用
            #     s = self.plugin(feature[:B//2], feature2[B//2:]) ## ! 要求输入x是concat的数据
            #     return feature, feature2, s
                return feature, feature2
            else:
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


class SimpleResidual_split_Backbone(nn.Module):

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112):
        super(SimpleResidual_split_Backbone, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes
        self.split_rate = split_rate

        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier') # 56 i:55/57 o:28/29
        self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0])

        self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')# 28 i:28/29 o:14/15
        self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1])

        self.conv3_up = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier') # 14 i:13 o:7 i:14/15 o:7/8
        self.layer3_up = self._make_layer(256, base_layer=block,layers_num=layers[2])

        self.conv3_down = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier') # 14 i:15 o:8
        self.layer3_down = self._make_layer(256, base_layer=block,layers_num=layers[2])

        self.conv4_up = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')# 7 i:7 o:4
        self.layer4_up = self._make_layer(512, base_layer=block,layers_num=layers[3])

        self.conv4_down = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')# 7 i:8 o:4
        self.layer4_down = self._make_layer(512, base_layer=block,layers_num=layers[3])

        self.fc5_up = nn.Linear(512*7*4, 256)
        self.fc5_down = nn.Linear(512*7*4, 256)

        # self.drop_out_up = Dropout2d(p=0.1)

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
        
        B,C,H,W = layer2.shape  
        
        conv3_up = self.conv3_up(layer2[:,:,:int(H*self.split_rate)-1, :])
        layer3_up = self.layer3_up(conv3_up)
        conv4_up = self.conv4_up(layer3_up)
        layer4_up = self.layer4_up(conv4_up)

        # layer4_up = self.drop_out_up(layer4_up.permute(2,3,0,1)).permute(2,3,0,1)
        # layer4_up = self.drop_out_up(layer4_up)

        layer4_up = layer4_up.view(layer4_up.size(0), -1)
        feature_up = self.fc5_up(layer4_up)

        conv3_down = self.conv3_down(layer2[:,:,int(H*self.split_rate)-1:, :])
        layer3_down = self.layer3_down(conv3_down)
        conv4_down = self.conv4_down(layer3_down)
        layer4_down = self.layer4_down(conv4_down)

        layer4_down = layer4_down.view(layer4_down.size(0), -1)
        feature_down = self.fc5_down(layer4_down)

        feature_up = torch.nn.functional.normalize(feature_up)
        feature_down = torch.nn.functional.normalize(feature_down)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature_up, feature_down


    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)
    

class SimpleResidual_split_attention_Backbone(SimpleResidual_split_Backbone):

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112):
        super(SimpleResidual_split_attention_Backbone, self).__init__(layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112)
        self.attention3 = AttentionModule(kernel_size=3, channels=256)
        self.attention4 = AttentionModule(kernel_size=3, channels=512)
        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        conv1 = self.conv1(x)
        layer1 = self.layer1(conv1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        
        B,C,H,W = layer2.shape  
        
        conv3_up = self.conv3_up(layer2[:,:,:int(H*self.split_rate)-1, :])
        layer3_up = self.layer3_up(conv3_up)

        attention3 = self.attention3(torch.cat((conv3_up, layer3_up), dim=1))
        # attention3 = self.attention3(torch.cat((conv3_up, layer3_up), dim=1))
        layer3_up  = attention3[0]*layer3_up

        conv4_up = self.conv4_up(layer3_up)
        layer4_up = self.layer4_up(conv4_up)

        attention4 = self.attention4(torch.cat((conv4_up, layer4_up)), dim=1)
        # attention4 = self.attention4(torch.cat((conv4_up, layer4_up)), dim=1)
        layer4_up  = attention4[0]*layer4_up

        layer4_up = self.drop_out_up(layer4_up.permute(2,3,0,1)).permute(2,3,0,1)

        layer4_up = layer4_up.view(layer4_up.size(0), -1)
        feature_up = self.fc5_up(layer4_up)

        conv3_down = self.conv3_down(layer2[:,:,int(H*self.split_rate)-1:, :])
        layer3_down = self.layer3_down(conv3_down)
        conv4_down = self.conv4_down(layer3_down)
        layer4_down = self.layer4_down(conv4_down)

        layer4_down = layer4_down.view(layer4_down.size(0), -1)
        feature_down = self.fc5_down(layer4_down)

        feature_up = torch.nn.functional.normalize(feature_up)
        feature_down = torch.nn.functional.normalize(feature_down)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature_up, feature_down


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

        ## mode 1
        # self.module_list.append(self.layer4)

        # if self.num_module>1:
        #     self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[-2])
        #     self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        #     self.module_list.append(self.conv4)
        #     self.module_list.append(self.layer3)

        # if self.num_module>2:
        #     self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[-3])
        #     self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier')
        #     self.module_list.append(self.conv3)
        #     self.module_list.append(self.layer2)
        # if self.num_module>3:
            
        #     self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[-4])
        #     self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')
        #     self.module_list.append(self.conv2)
        #     self.module_list.append(self.layer1)
        
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
    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112, SSS_location=4, three_branch=False,
                 cl=False, ll=False):
        #sss_location: 1~4
        super(SimpleResidual_split_abla, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes
        self.split_rate = split_rate
        self.three_branch = three_branch
        self.cl=cl
        self.ll=ll
        self.module_list = []
        
        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier') # 56 i:55/57 o:28/29
        self.module_list.append(self.conv1)

        # mode1
        # if SSS_location>1:
        #     self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0])
        #     self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')# 28 i:28/29 o:14/15
        #     self.module_list.append(self.layer1)
        #     self.module_list.append(self.conv2)
        # if SSS_location>2:
        #     self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1])
        #     self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier') # 14 i:13 o:7 i:14/15 o:7/8
        #     self.module_list.append(self.layer2)
        #     self.module_list.append(self.conv3)
        # if SSS_location>3:
        #     self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[2])
        #     self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        #     self.module_list.append(self.layer3)
        #     self.module_list.append(self.conv4)

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
        # if SSS_location==4:
        #     self.fc_up = nn.Linear(512*7*3, 256) # last:3 # TODO
        #     self.fc_down = nn.Linear(512*7*4, 256)# last:4
        # else:
        #     self.fc_up = nn.Linear(512*7*4, 256) # last:3 # TODO
        #     self.fc_down = nn.Linear(512*7*4, 256)# last:4
        self.fc_up = nn.Linear(512*7*7, 256)
        self.fc_down = nn.Linear(512*7*7, 256)
        if self.three_branch:
            self.SSS_up_mask = SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block)
            if SSS_location==4:
                self.fc_up_mask = nn.Linear(512*7*3, 256) # last:3 # TODO
            else:
                self.fc_up_mask = nn.Linear(512*7*4, 256) # last:3 # TODO
        
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
    
    def forward(self, x, classes=None):
        
        out = self.backbone(x)
        B,C,H,W = out.shape
        if classes is not None:
            down_index = classes<1
            # print("after bool:", down_index.size())
            # down_index = down_index.unsqueeze(out.dim()).expand_as(out)
            down_input = out[down_index][:,:,int(H*self.split_rate)-1:, :]
            # print("after bool as index:", down_input.size())
            if self.three_branch:
                up_index = classes>0
                # up_index = up_index.unsqueeze(up_index.dim()).expand_as(out)
                up_input = out[down_index][:,:,:int(H*self.split_rate)-1, :]
                up_input_mask = out[up_index][:,:,:int(H*self.split_rate)-1, :]
                SSS_out_up_mask = self.SSS_up_mask(up_input_mask)
                SSS_out_up_mask = SSS_out_up_mask.view(SSS_out_up_mask.size(0), -1)
                feature_up_mask = self.fc_up_mask(SSS_out_up_mask)
                feature_up_mask = torch.nn.functional.normalize(feature_up_mask)
                
            else:
                up_input = out[:,:,:int(H*self.split_rate)-1, :]
        else: # 方便测试
            # up_input = out[:,:,:int(H*self.split_rate)-1, :]
            # down_input = out[:,:,int(H*self.split_rate)-1:, :]
            # up_input = out[:,:,:, :int(W*self.split_rate)-1]
            # down_input = out[:,:,:, int(W*self.split_rate)-1:]
            up_input = out
            down_input = out
            if self.three_branch:
                SSS_out_up_mask = self.SSS_up_mask(up_input)
                SSS_out_up_mask = SSS_out_up_mask.view(SSS_out_up_mask.size(0), -1)
                feature_up_mask = self.fc_up_mask(SSS_out_up_mask)
                feature_up_mask = torch.nn.functional.normalize(feature_up_mask)         
        if not self.cl and not self.ll:
            SSS_out_up = self.SSS_up(up_input)
            SSS_out_down = self.SSS_down(down_input)
        else:
            SSS_out_up_conv4 = self.SSS_up.subject[1](up_input)
            SSS_out_down_conv4 = self.SSS_down.subject[1](down_input)
            if self.cl:
                SSS_out_up = self.SSS_up.subject[0](SSS_out_up_conv4+SSS_out_down_conv4)
                SSS_out_down = self.SSS_down.subject[0](SSS_out_down_conv4+SSS_out_up_conv4)
            else:
                SSS_out_up = self.SSS_up.subject[0](SSS_out_up_conv4)
                SSS_out_down = self.SSS_down.subject[0](SSS_out_down_conv4)
            if self.ll:
                SSS_out_up += SSS_out_down
                SSS_out_down += SSS_out_up
        # SSS_out_up = self.drop_out_up(SSS_out_up.permute(2,3,0,1)).permute(2,3,0,1)
        # SSS_out_up = self.drop_out_up(SSS_out_up)

        SSS_out_up = SSS_out_up.view(SSS_out_up.size(0), -1)
        SSS_out_down = SSS_out_down.view(SSS_out_down.size(0), -1)

        feature_up = self.fc_up(SSS_out_up)
        feature_down = self.fc_down(SSS_out_down)
        
        feature_up = torch.nn.functional.normalize(feature_up)
        feature_down = torch.nn.functional.normalize(feature_down)
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        if self.three_branch:
            return feature_up, feature_down, feature_up_mask
        else:
            return feature_up, feature_down


    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class SimpleResidual_split_abla_fusion(SimpleResidual_split_abla):
    def __init__(self, layers, fusion_mode=None, loss_down=None, fusion_loss=None, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112, SSS_location=4, three_branch=False):
        #sss_location: 1~4
        super(SimpleResidual_split_abla_fusion, self).__init__(layers=layers, use_se=use_se, block=block, phase=phase, input_size=input_size, split_rate=split_rate, SSS_location=SSS_location, three_branch=three_branch)
        self.fusion_mode = fusion_mode
        
        self.loss_down = loss_down
        self.loss = fusion_loss
        assert (loss_down is not None and fusion_loss is not None) or (loss_down is  None and fusion_loss is  None) # 要么都有，要么都没有
        
        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    nn.init.constant_(m.bias, 0.0)
                    torch.nn.init.xavier_uniform_(m.weight.data)
        
    def forward(self, x, classes=None, ids=None):
        # assert not(self.margin_loss_down is not None and id is not None) # 两者异或会出问题
        out = self.backbone(x)
        B,C,H,W = out.shape
        if classes is not None:
            down_index = classes<1
            # print("after bool:", down_index.size())
            # down_index = down_index.unsqueeze(out.dim()).expand_as(out)
            if self.loss is not None:
                down_input = out[:,:,int(H*self.split_rate)-1:, :]
            else:
                down_input = out[down_index][:,:,int(H*self.split_rate)-1:, :]
            # print("after bool as index:", down_input.size())
            if self.three_branch:
                up_index = classes>0
                # up_index = up_index.unsqueeze(up_index.dim()).expand_as(out)
                up_input = out[down_index][:,:,:int(H*self.split_rate)-1, :]
                up_input_mask = out[up_index][:,:,:int(H*self.split_rate)-1, :]
                SSS_out_up_mask = self.SSS_up_mask(up_input_mask)
                SSS_out_up_mask = SSS_out_up_mask.view(SSS_out_up_mask.size(0), -1)
                feature_up_mask = self.fc_up_mask(SSS_out_up_mask)
                feature_up_mask = torch.nn.functional.normalize(feature_up_mask)
                
            else:
                up_input = out[:,:,:int(H*self.split_rate)-1, :]
        else:
            if self.three_branch:
                SSS_out_up_mask = self.SSS_up_mask(up_input)
                SSS_out_up_mask = SSS_out_up_mask.view(SSS_out_up_mask.size(0), -1)
                feature_up_mask = self.fc_up_mask(SSS_out_up_mask)
                feature_up_mask = torch.nn.functional.normalize(feature_up_mask)
            up_input = out[:,:,:int(H*self.split_rate)-1, :]
            down_input = out[:,:,int(H*self.split_rate)-1:, :]
            
        SSS_out_up = self.SSS_up(up_input)
        SSS_out_down = self.SSS_down(down_input)
        # SSS_out_up = self.SSS_up(out[:,:,:int(H*self.split_rate)-1, :]) # int(H*self.split_rate)-1
        # SSS_out_up = self.SSS_up(out/2) # int(H*self.split_rate)-1
        # SSS_out_down = self.SSS_down(out[:,:,int(H*self.split_rate)-1:, :])
        # SSS_out_down = self.SSS_down(out/2)

        # SSS_out_up = self.drop_out_up(SSS_out_up.permute(2,3,0,1)).permute(2,3,0,1)
        # SSS_out_up = self.drop_out_up(SSS_out_up)

        SSS_out_up = SSS_out_up.view(SSS_out_up.size(0), -1)
        SSS_out_down = SSS_out_down.view(SSS_out_down.size(0), -1)

        feature_up = self.fc_up(SSS_out_up)
        feature_down = self.fc_down(SSS_out_down)
        
        feature_up = torch.nn.functional.normalize(feature_up)
        feature_down = torch.nn.functional.normalize(feature_down)
        
        if self.loss is not None:
            assert ids is not None
            feature_down[classes>0]=self.loss_down.weight[ids[classes>0].long()]
            outputs_down, original_logits_down = self.loss_down(feature_down[classes<1], ids[classes<1])

            outputs, original_logits = self.loss(self.fusion_mode(feature_up, feature_down), ids)
            return feature_up, outputs, original_logits, outputs_down, original_logits_down
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        if self.three_branch:
            return feature_up, feature_down, feature_up_mask
        else:
            return feature_up, feature_down


class SimpleResidual_split_abla_num(SimpleResidual_split_abla):
    def __init__(self, layers, num_classifier=2, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112, SSS_location=4, three_branch=False):
        #sss_location: 1~4
        super(SimpleResidual_split_abla_num, self).__init__(layers=layers, use_se=use_se, block=block, phase=phase, input_size=input_size, split_rate=split_rate, SSS_location=SSS_location, three_branch=three_branch)
        self.num_classifier = num_classifier
        branch_list = []
        fc_list = []
        for i in range(num_classifier):
            branch_list.append(SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block))
            fc_list.append(nn.Linear(512*7*3, 512//self.num_classifier))
        self.branch_list = nn.ModuleList(branch_list)
    
class SimpleResnet_atten_split_abla(nn.Module):
    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], split_rate=56.7366/112, SSS_location=4):
        #sss_location: 1~4
        super(SimpleResnet_atten_split_abla, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes
        self.split_rate = split_rate
        self.module_list = []
        
        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier') # 56 i:55/57 o:28/29
        self.module_list.append(self.conv1)

        # mode1
        # if SSS_location>1:
        #     self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0])
        #     self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')# 28 i:28/29 o:14/15
        #     self.module_list.append(self.layer1)
        #     self.module_list.append(self.conv2)
        # if SSS_location>2:
        #     self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1])
        #     self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier') # 14 i:13 o:7 i:14/15 o:7/8
        #     self.module_list.append(self.layer2)
        #     self.module_list.append(self.conv3)
        # if SSS_location>3:
        #     self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[2])
        #     self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        #     self.module_list.append(self.layer3)
        #     self.module_list.append(self.conv4)

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

        self.atten = AttentionModule(kernel_size=3, pool_size = (1, 2, 3), channels=256)
        
        self.SSS_up = SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block)
        self.SSS_down = SpaceSliceStrategy_subject(layers=layers[SSS_location-1:], block=block)

        # self.drop_out_up = Dropout2d(p=0.1)
        self.fc_up = nn.Linear(512*7*7, 256) # last:3 # TODO
        self.fc_down = nn.Linear(512*7*7, 256)# last:4

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

        atten_out_up, atten_out_down = self.atten(out)  
        
        SSS_out_up = self.SSS_up(atten_out_up) # int(H*self.split_rate)-1
        SSS_out_down = self.SSS_down(atten_out_down)

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


class SimpleResidualBackbone_smask(nn.Module):

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='test', input_size=[112, 112]):
        super(SimpleResidualBackbone_smask, self).__init__()
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
        return nn.Sequential(*layers)


    def forward(self, x, ldm):
        conv1 = self.conv1(x)

        mask = torch.ones(conv1_1.shape).to('cuda')*0.5 # batch c w h
        # print(ldm.shape)
        # ldm # batch 1 10
        # dist_em = ldm[:, 0, 5]-ldm[:, 0, 1]
        xmax = conv1_1.shape[2]
        ymax = ( ldm[:, 1]+ldm[:, 5] )/2*xmax/112
        # print(mask[0, 0, 0:xmax, 0:ymax])
        # mask[:, :, 0:xmax, 0:ymax] = 1
        for b in range(conv1_1.shape[0]):
            mask[b, :, 0:ymax[b], 0:xmax] = 1
        
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


class SimpleResidualBackbone_multiTask(nn.Module):

    def __init__(self, layers, use_se=False, block=SimpleResidualUnit, phase='train', input_size=[112, 112], att_type=None):
        super(SimpleResidualBackbone_multiTask, self).__init__()
        self.phase = phase
        self.use_se = use_se
        # self.num_classes = num_classes

        self.conv1 = ConvPrelu(3, 64, kernel_size=3, stride=2, padding=1, filter='xavier')

        if att_type=='BAM':
            self.bam1 = BAM(64)
            self.bam2 = BAM(128)
            self.bam3 = BAM(256)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(64, base_layer=block,layers_num=layers[0], att_type=att_type)

        self.conv2 = ConvPrelu(64, 128, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer2 = self._make_layer(128, base_layer=block,layers_num=layers[1], att_type=att_type)

        self.conv3 = ConvPrelu(128, 256, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer3 = self._make_layer(256, base_layer=block,layers_num=layers[2], att_type=att_type)

        self.conv4 = ConvPrelu(256, 512, kernel_size=3, stride=2, padding=1, filter='xavier')
        self.layer4 = self._make_layer(512, base_layer=block,layers_num=layers[3], att_type=att_type)

        self.fc5 = nn.Linear(512*7*7, 512)
        self.fc5_mask = nn.Linear(512*7*7, 512)
        self.prelu_mask = nn.PReLU(num_parameters=512, init=0.25)
        self.mask_classify = nn.Linear(512,2, bias=False)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Linear):
                    # m.bias.data.fill_(0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                        torch.nn.init.xavier_uniform_(m.weight.data)

    def _make_layer(self, in_channels, base_layer, layers_num, att_type):
        layers=[]
        for i in range(layers_num):
            layers.append(base_layer(in_channels, use_se=self.use_se, use_cbam=att_type=='CBAM'))
            self.use_se = False# TODO
        return nn.Sequential(*layers)


    def forward(self, x):
        conv1 = self.conv1(x)
        layer1 = self.layer1(conv1)
        if not self.bam1 is None:
            layer1 = self.bam1(layer1)
        conv2 = self.conv2(layer1)
        layer2 = self.layer2(conv2)
        if not self.bam2 is None:
            layer2 = self.bam2(layer2)
        conv3 = self.conv3(layer2)
        layer3 = self.layer3(conv3)
        if not self.bam3 is None:
            layer3 = self.bam3(layer3)
        conv4 = self.conv4(layer3)
        layer4 = self.layer4(conv4)
        layer4 = layer4.view(layer4.size(0), -1)

        feature = self.fc5(layer4)
        feature = torch.nn.functional.normalize(feature)
        feature_mask = self.fc5_mask(layer4)
        feature_mask = torch.nn.functional.normalize(feature_mask)
        # feature_mask = self.prelu_mask(feature_mask)
        # feature_mask = self.mask_classify(feature_mask) 
        # print("backbone device: %d is working"%(torch.cuda.current_device()))
        return feature, feature_mask

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


def SimpleResnet_split_64(**kwargs):
    """Constructs a SimpleResnet_split_64 model.
    """
    model = SimpleResidual_split_Backbone([3, 8, 16, 3], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_split_36(**kwargs):
    """Constructs a SimpleResnet_split_36 model.
    """
    model = SimpleResidual_split_Backbone([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
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


def SimpleResnet_split_abla_36_fusion(**kwargs):
    """Constructs a SimpleResnet_fullsplit_36 model.
    """
    model = SimpleResidual_split_abla_fusion([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model

def SimpleResnet_atten_split_abla_36(**kwargs):
    """Constructs a SimpleResnet_fullsplit_36 model.
    """
    model = SimpleResnet_atten_split_abla([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_split_abla_64(**kwargs):
    """Constructs a SimpleResnet_fullsplit_36 model.
    """
    model = SimpleResidual_split_abla([3, 8, 16, 3], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model


def SimpleResnet_36_multiTask(**kwargs):
    """Constructs a SimpleResnet_split_36 model.
    """
    model = SimpleResidualBackbone_multiTask([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model

def SimpleResnet_split_attention_36(**kwargs):
    """Constructs a SimpleResnet_fullsplit_36 model.
    """
    model = SimpleResidual_split_attention_Backbone([2, 4, 8, 2], use_se=False, block=SimpleResidualUnit, **kwargs)
    return model