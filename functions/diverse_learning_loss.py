import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class DiverseLearnLoss(nn.Module):
    def __init__(self, t=1, scale=1, branch=1, depth=3, batch_size=512):
        '''
        @t: loss = |t-A*B|
        @scale: 尺寸金字塔
        @branch: attention的重复次数
        @depth: 特征金字塔
        '''
        super(DiverseLearnLoss, self).__init__()
        self.t = t
        self.branch = branch
        self.scale = scale
        self.depth = depth
        self.batch_size = batch_size
        # self.mask = self.get_mask(batch_size).to("cuda")
    
    # def forward(self, mask_list):
    #     '''
    #     @mask_list: list D*S*B*N*1*W*H
    #     '''
    #     loss = 0
    #     for i in range(self.depth):
    #         for j in range(self.scale):
    #             ds_mask_list = mask_list[i][j] # BR*N*1*W*H
    #             ds_mask_list = torch.squeeze(ds_mask_list, dim=-3)
    #             mask_dist = torch.unsqueeze(mask_list, dim=1) - mask_list # 期望为 B* B *N* W*H
    #             mask_dist = torch.norm(mask_dist.permute([2,0,1,3,4]), p=2, dim=[3,4]) # N*B* B
    #             # print(mask_dist.size())
    #             mask_score = self.t-mask_dist[(self.t-mask_dist)>0]
    #             # print(mask_score.size())
    #             mask_score = torch.sum(mask_score) - self.t*self.branch*self.batch_size
    #             # print(mask_score)
    #             if self.branch!=1:
    #                 mask_score = mask_score/self.branch/(self.branch-1)

    #             loss = loss + mask_score

    #     loss = loss / self.batch_size
    #     return loss
    
    def forward(self, mask_list):
        '''
        @mask_list: list D*S*B*N*1*W*H
        '''
        loss = 0
        num = self.branch*self.scale*self.depth
        mask_list = torch.squeeze(mask_list, dim=-3)# 去掉通道
        mask_list = mask_list.view(num, self.batch_size, mask_list.size(-2), mask_list.size(-1))# 通道整理
        # mask_list = mask_list.view(num, self.batch_size, -1)

        # for i in range(self.batch_size):
        #     mask_i_list = mask_list[i]
        #     mask_dist = mask_i_list.mul(mask_i_list.t())
        #     mask_score = torch.norm(mask_dist, p=1)
        #     mask_score = (mask_score - num)/2# 因为没有归一化，所以减去对角线不对
        #     mask_score = mask_score/num/(num+1)*2
        #     loss = loss + mask_score
        # 等价为
        # mask_dist = torch.bmm(mask_list, mask_list.permute([0,2,1])) # N * (S*B*D)* (S*B*D) # 这里的bmm和matmul效果一样
        # # mask_score = torch.sum(mask_dist, [1,2])
        # mask_score = torch.norm(mask_dist, p=1, dim=[1,2])
        # mask_score = (mask_score - num)/2# 因为没有归一化，所以减去对角线不对
        # mask_score = mask_score/num/(num+1)*2
        # loss = torch.sum(mask_score, dim=0)
        # print(mask_list.size())
        # 更改为
        mask_dist = torch.unsqueeze(mask_list, dim=1) - mask_list # 期望为 (S*B*D)* (S*B*D) *N* W*H 遍历相减
        # mask_dist = mask_list.expand(self.batch_size, num, num, mask_list.size(2)) - mask_list # 期望为 N * (S*B*D)* (S*B*D) * (W*H)
        # print(mask_dist.size())
        mask_dist = torch.norm(mask_dist.permute([2,0,1,3,4]), p=2, dim=[3,4]) # N*(S*B*D)* (S*B*D)
        # print(mask_dist)
        # print(mask_dist.size())
        mask_score = self.t-mask_dist[(self.t-mask_dist)>0]
        # print(mask_score.size())
        mask_score = torch.sum(mask_score) - self.t*num*self.batch_size
        # print(mask_score)
        if num!=1:
            mask_score = mask_score/num/(num-1)
        loss = mask_score

        loss = loss / self.batch_size

        return loss