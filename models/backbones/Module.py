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
        
## 以下代码全部来自xcos开源的代码 https://github.com/ntubiolin/xcos/blob/d5bf9be562f7dfe3e0ae3972b33ef3ac5c6cc691/src/model/model.py#L43
cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def l2normalize(x):
    return nn.function.normalize(x, p=2, dim=1)


class FrobeniusInnerProduct(nn.Module):
    def __init__(self):
        super(FrobeniusInnerProduct, self).__init__()

    def forward(self, grid_cos_map, attention_map):
        """ Compute the Frobenius inner product
            with grid cosine map and attention map.
        Args:
            grid_cos_map (Tensor of size([bs, 7, 7, 1]))
            attention_map (Tensor of size([bs, 7, 7, 1])
        Returns:
            Tensor of size [bs, 1]: aka. xCos values
        """
        attentioned_gird_cos = (grid_cos_map * attention_map)
        # attentioned_gird_cos:  torch.Size([bs, 7, 7, 1]) ->[bs, 49]
        attentioned_gird_cos = attentioned_gird_cos.view(attentioned_gird_cos.size(0), -1)
        frobenius_inner_product = attentioned_gird_cos.sum(1)
        return frobenius_inner_product

class GridCos(nn.Module):
    def __init__(self):
        super(GridCos, self).__init__()

    def forward(self, feat_grid_1, feat_grid_2):
        """ Compute the grid cos map with 2 input conv features
        Args:
            feat_grid_1 ([type]): [description]
            feat_grid_2 ([type]): [description]
        Returns:
            Tensor of size([bs, 7, 7, 1]: [description]
        """
        feat_grid_1 = feat_grid_1.permute(0, 2, 3, 1)  # CHW to HWC
        feat_grid_2 = feat_grid_2.permute(0, 2, 3, 1)
        output_size = feat_grid_1.size()[0:3] + torch.Size([1])

        feat1 = feat_grid_1.contiguous().view(-1, feat_grid_1.size(3))
        feat2 = feat_grid_2.contiguous().view(-1, feat_grid_2.size(3))
        feat1 = l2normalize(feat1)
        feat2 = l2normalize(feat2)
        grid_cos_map = cos(feat1, feat2).view(output_size)
        return grid_cos_map


class XCosAttention(nn.Module):
    def __init__(self, use_softmax=True, use_sigmoid=False, softmax_t=1, chw2hwc=True):
        super(XCosAttention, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU())
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.name = 'AttenCosNet'
        self.USE_SOFTMAX = use_softmax
        self.USE_SIGMOID = use_sigmoid
        self.SOFTMAX_T = softmax_t
        self.chw2hwc = chw2hwc
        
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.02) # 自己的std是0.01，xcos作者的是0.02
                m.bias.data.zero_() # TODO 需要验证

    def softmax(self, x, T=1):
        x /= T
        return torch.nn.functional.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)
    
    def sigmoid(self, x):
        return torch.nn.functional.sigmoid(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)
    
    def divByNorm(self, x):
        '''
            attention_weights.size(): [bs, 1, 7, 7]
        '''
        x -= x.view(x.size(0),
                    x.size(1), -1).min(dim=2)[0].repeat(1,
                                                        1,
                                                        x.size(2) * x.size(3)).view(x.size(0),
                                                                                    x.size(1),
                                                                                    x.size(2),
                                                                                    x.size(3))
        x /= x.view(x.size(0),
                    x.size(1), -1).sum(dim=2).repeat(1,
                                                     1,
                                                     x.size(2) * x.size(3)).view(x.size(0),
                                                                                 x.size(1),
                                                                                 x.size(2),
                                                                                 x.size(3))
        return x

    def forward(self, feat_grid_1, feat_grid_2):
        '''
            feat_grid_1.size(): [bs, 32, 7, 7]
            attention_weights.size(): [bs, 1, 7, 7]
        '''
        # XXX Do I need to normalize grid_feat?
        conv1 = self.embedding_net(feat_grid_1)
        conv2 = self.embedding_net(feat_grid_2)
        fused_feat = torch.cat((conv1, conv2), dim=1)
        attention_weights = self.attention(fused_feat)
        # To Normalize attention
        if self.USE_SOFTMAX:
            attention_weights = self.softmax(attention_weights, self.SOFTMAX_T)
        elif self.USE_SIGMOID:
            attention_weights = self.sigmoid(attention_weights)
        else:
            attention_weights = self.divByNorm(attention_weights)
        if self.chw2hwc:
            attention_weights = attention_weights.permute(0, 2, 3, 1)
        return attention_weights

class PCM_AM(nn.Module):
    def __init__(self, ):
        super(PCM_AM, self).__init__()
        self.pcm = GridCos()
        self.am = XCosAttention(use_sigmoid=True, chw2hwc=False)
        self.score = FrobeniusInnerProduct()

    def forward(self, x, y):
        '''
        @x,y : Batch*32*7*7 未被归一化的特征
        '''
        cosScore = self.pcm(x, y)
        mask = self.am(x, y)
        S = self.score(cosScore, mask)/49
        
        return S
        

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