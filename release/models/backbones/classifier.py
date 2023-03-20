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
        
        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        #nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)

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
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_loit)
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