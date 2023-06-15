import torch.nn as nn
import torch

## 以下代码全部来自xcos开源的代码 https://github.com/ntubiolin/xcos/blob/d5bf9be562f7dfe3e0ae3972b33ef3ac5c6cc691/src/model/model.py#L43
cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def l2normalize(x):
    return nn.functional.normalize(x, p=2, dim=1)


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
                m.bias.data.zero_()

    def softmax(self, x, T=1):
        x /= T
        return torch.nn.functional.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)
    
    def sigmoid(self, x):
        return torch.sigmoid(x).view_as(x)
    
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
        self.am = XCosAttention(use_sigmoid=True, chw2hwc=False, use_softmax=False)
        self.score = FrobeniusInnerProduct()

    def forward(self, x, y):
        '''
        @x,y : Batch*32*7*7 未被归一化的特征
        '''
        cosScore = self.pcm(x, y)
        mask = self.am(x, y)
        S = self.score(cosScore, mask)/49
        
        return S