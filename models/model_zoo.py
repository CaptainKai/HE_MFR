# pylint: disable=wildcard-import, unused-wildcard-import

from .backbones import *

__all__ = ['models_list', 'get_model']
def cat(x,y):
    return torch.cat([x,y], 1)
models_list = {
    # simple resnet
    'SimpleResnet_20': SimpleResnet_20,
    'SimpleResnet_36': SimpleResnet_36,
    'SimpleResnet_64': SimpleResnet_64,
    # simple resnet se
    'SimpleResnet_20_Se': SimpleResnet_20_Se,
    'SimpleResnet_36_Se': SimpleResnet_36_Se,
    'SimpleResnet_64_Se': SimpleResnet_64_Se,
    # simple resnet split
    'SimpleResnet_split_36': SimpleResnet_split_36,
    'SimpleResnet_split_64': SimpleResnet_split_64,
    'SimpleResnet_fullsplit_36': SimpleResnet_fullsplit_36,
    'SimpleResnet_split_abla_36': SimpleResnet_split_abla_36,
    'SimpleResnet_split_abla_64': SimpleResnet_split_abla_64,
    'SimpleResnet_atten_split_abla_36': SimpleResnet_atten_split_abla_36,
    'SimpleResnet_split_attention_36': SimpleResnet_split_attention_36,
    'SimpleResnet_36_multiTask': SimpleResnet_36_multiTask,
    'SimpleResnet_split_abla_36_fusion': SimpleResnet_split_abla_36_fusion,
    # resnet
    'resnet18': ResNet_18,
    'resnet34': ResNet_34,
    'resnet50': ResNet_50,
    'resnet101': ResNet_101,
    'resnet152': ResNet_152,
    # split
    'resnet50_split': ResNet_50_split,
    'ResNet50_split_abla': ResNet_50_split_abla,
    # split_attention
    'resnet50_split_attention': ResNet_50_split_attention,
    'resnet50_split_attention_abla': ResNet_50_split_attention_abla,
    # resnest
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnest200': resnest200,
    'resnest269': resnest269,
    # senet
    'se_resnet_18': se_resnet_18,
    'se_resnet_34': se_resnet_34,
    'se_resnet_50': se_resnet_50,
    'se_resnet_101': se_resnet_101,
    'se_resnet_152': se_resnet_152,
    # se resnext
    'se_resnext_50': se_resnext_50,
    'se_resnext_101': se_resnext_101,
    'se_resnext_152': se_resnext_152,
    # densenet
    'densenet_121': DenseNet_121,
    'densenet_161': DenseNet_161,
    'densenet_169': DenseNet_169,
    'densenet_201': DenseNet_201,
    # HPDA
    "HPDA_res50": HPDA_res50,
    # classifier
    'MarginCosineProduct': MarginCosineProduct,
    'ArcFace': ArcFace,
    'CurricularFace': CurricularFace,
    'DCFace': DCFace,
    'MarginCosineProduct_sub': MarginCosineProduct_sub,
    'ASoftmax': ASoftmax,
    'Sphereface2': Sphereface2,
    # mapping network
    'MappingNetwork_8': mapping_net_8,
    'MappingNetwork_4': mapping_net_4,
    'MappingNetwork_2': mapping_net_2,
    'MappingNetwork_1': mapping_net_1,
    # fusion
    'MLB': MLB,
    'ADD': torch.add,
    'CAT': cat,
    ## attention
    "PCM_AM": PCM_AM,
    
}

# model_list = list(models.keys())

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.

    Returns
    -------
    Module:
        The model.
    """
    if name not in models_list.keys():
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models_list.keys()))))
    net = models_list[name](**kwargs)
    return net
