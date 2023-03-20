import torch.nn as nn


class MappingNet(nn.Module):

    def __init__(self, inchannel, outchannel, layer_num=8):
        super(MappingNet, self).__init__()
        self.layer_num = layer_num
        self.net = self._make_layer(inchannel, outchannel)
        
    
    def _make_layer(self, inchannel, outchannel):
        layers = []
        
        for i in range(self.layer_num):
            layers.append(nn.Linear(inchannel, outchannel))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        return out

    def save(self, file_path):
        import torch
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

def mapping_net_8(inchannel, outchannel):
    network = MappingNet(inchannel, outchannel, 8)
    return network

def mapping_net_4(inchannel, outchannel):
    network = MappingNet(inchannel, outchannel, 4)
    return network

def mapping_net_2(inchannel, outchannel):
    network = MappingNet(inchannel, outchannel, 2)
    return network

def mapping_net_1(inchannel, outchannel):
    network = MappingNet(inchannel, outchannel, 1)
    return network
