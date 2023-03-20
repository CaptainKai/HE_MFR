import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter 

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)# 张量积计算，并限制最小值


class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=-0.35): # s=64, m=-0.5
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # # sphere Norm
        # self.weight = torch.Tensor(out_features, in_features)
        # nn.init.normal_(self.weight, mean=0.0, std=1.0)
        # self.weight = self.weight/torch.norm(self.weight, 2, 1, True) # error dim match
        # self.weight = Parameter(self.weight)
        
        # self.weight = F.normalize(self.weight) # error floattensor not parameter
        # with torch.no_grad():
        #     self.weight = torch.div(self.weight, torch.norm(self.weight, 2, 1, True))
        # print("device: %d"%(torch.cuda.current_device()))
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        # print(self.weight.shape)# class*512
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)# 512 one-hot:独热编码 任意时候只有一位有效
        one_hot.scatter_(1, label.view(-1, 1), 1.0)# [0,gt]=1
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine + one_hot * self.m)
        # print("amsoft device: %d is working"%(torch.cuda.current_device()))


        return output, self.s *cosine
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

import math

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, s = 64.0, m = 0.50, easy_margin = False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        
        # self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        # #nn.init.xavier_uniform_(self.kernel)
        # nn.init.normal_(self.kernel, std=0.01)
        self.kernel = torch.Tensor(in_features, out_features)
        nn.init.normal_(self.kernel, mean=0.0, std=1.0)
        self.kernel = l2_norm(self.kernel, axis = 0) # error dim match
        self.kernel = Parameter(self.kernel)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class DCFace(nn.Module):
    def __init__(self, in_features, out_features, s=[30.0,64.0], m=[-0.35,0.5], type_index=0): # s=64, m=-0.5
        super(DCFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s[type_index]
        self.m = m[type_index]
        self.weight_up = Parameter(torch.Tensor(out_features, in_features))
        self.weight_down = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_up)
        nn.init.xavier_uniform_(self.weight_down)
        self.type_index = type_index

    def forward(self, input_up, input_down, label, extra_label=None):
        cosine_up = cosine_sim(input_up, self.weight_up)
        cosine_down = cosine_sim(input_down, self.weight_down)
        if extra_label is not None:
            cosine_down = cosine_down*(1-extra_label.unsqueeze(1))
            cosine = ( cosine_up + cosine_down + cosine_up*(extra_label.unsqueeze(1)) )/2
        else:
            cosine = (cosine_up+cosine_down)/2
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)# 512 one-hot:独热编码 任意时候只有一位有效
        one_hot.scatter_(1, label.view(-1, 1), 1.0)# [0,gt]=1
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine + one_hot * self.m)
        # print("amsoft device: %d is working"%(torch.cuda.current_device()))


        return output, self.s *cosine

    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-8, min=0.1, average=True, tri=True): # 大于0.8的进行处理，大于。1
        super(CosineSimilarityLoss, self).__init__()
        self.dim = dim
        self.eps = eps
        self.min = min
        self.average = average
        self.tri = tri
    
    def forward(self, a, b, labels):
        x1 = a[labels.view(-1,1)[:,0], :]
        x2 = b[labels.view(-1,1)[:,0], :]
        # print(x1.shape,x2.shape)
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, self.dim)
        w2 = torch.norm(x2, 2, self.dim)
        cosine = ip / torch.ger(w1,w2).clamp(min=self.eps)# 张量积计算，并限制最小值
        # cosine = cosine
        if self.tri:
            cosine = torch.diag(cosine)
        # print(cosine.shape)
        # loss = 1 - cosine # TODO
        loss = cosine
        # loss = loss.clamp(min=self.min)-self.min
        # loss = torch.sum(loss)
        loss_filt = loss[loss>=self.min]
        loss_total = torch.sum(loss_filt)
        # print(loss)
        if self.average:
            loss_total = loss_total / max(1, loss_filt.shape[0])
        return loss_total


class MarginCosineProduct_sub(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=-0.35, sub_k=2): # s=64, m=-0.5
        super(MarginCosineProduct_sub, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub_k = sub_k
        self.weight = Parameter(torch.Tensor(out_features, in_features, self.sub_k))
        nn.init.xavier_uniform_(self.weight)
        # print("device: %d"%(torch.cuda.current_device()))
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine_result = []
        for i in range(self.sub_k):
            cosine_result.append(cosine_sim(input, self.weight[:,:,i]))
        # sub_k * B * class
        cosine_result = torch.stack(cosine_result, dim=0)
        # print(cosine_result.shape)
        cosine = torch.max(cosine_result, 0)[0]
        # print(cosine.shape)
        # print(self.weight.shape)# 512*weight
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)# Batch one-hot:独热编码 任意时候只有一位有效
        one_hot.scatter_(1, label.view(-1, 1), 1.0)# [0,gt]=1
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine + one_hot * self.m)
        # print("amsoft device: %d is working"%(torch.cuda.current_device()))


        return output, self.s *cosine
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss


class ASoftmax(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features): # s=64, m=-0.5
        super(ASoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # print("device: %d"%(torch.cuda.current_device()))
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)

        return cosine
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


class Sphereface2(nn.Module):
    def __init__(self, in_features, out_features, lamda=0.7, r=30, m=0.4, t=3, gFunc=None, b=0.6):
        '''
        lamda设置为0.7后，我的版本loss小很多
        '''
        super(Sphereface2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lamda = lamda
        self.r = r
        self.m = m
        self.t = t
        self.gFunc=gFunc
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.b = b
        # self.b = Parameter(torch.Tensor(in_features, 1))
        nn.init.xavier_uniform_(self.weight)
        # nn.init.xavier_uniform_(self.b)
        self.mlambda = [
            lambda x: 2 * ((x + 1) / 2) ** self.t - 1,
        ]
    def forward(self, input, label):
        # cosine = cosine_sim(input, self.weight)
        # # cos_theta = cosine_sim(input, self.weight)
        # if self.gFunc is not None:
        #     cosine = 2*((cosine+1)/2)**self.t-1
        # # print(cosine)
        # one_hot = torch.zeros_like(cosine)# 512 one-hot:独热编码 任意时候只有一位有效
        # one_hot.scatter_(1, label.view(-1, 1), 1.0)# [0,gt]=1
        # one = torch.ones_like(cosine)
        # one.scatter_(1, label.view(-1,1), -1.0)
        # loss = ((1-self.lamda)*one+one_hot) * torch.log(torch.exp(one * (self.r*cosine + self.b)+self.r*self.m) + 1)/self.r
        # # print(loss)
        # # print("loss:,",loss)
        # # print("loss_sum1,",loss.sum(dim=1))
        # # print("loss_sum2,", loss.sum(dim=0), loss.sum(dim=0).size())
        # # print("loss_ave", loss.sum(dim=1)/input.size(0))
        # return cosine, loss.sum(dim=1).mean()
        '''
        以下是git版本
        '''
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.r * (self.mlambda[0](cos_theta) - self.m) + self.b
        cos_m_theta1 = self.r * (self.mlambda[0](cos_theta) + self.m) + self.b
        cos_p_theta = (self.lamda / self.r) * torch.log(1 + torch.exp(-cos_m_theta))

        cos_n_theta = ((1 - self.lamda) / self.r) * torch.log(1 + torch.exp(cos_m_theta1))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        loss = (one_hot * cos_p_theta) + (1 - one_hot) * cos_n_theta
        loss = loss.sum(dim=1)
        
        return cos_theta, loss.mean()
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

