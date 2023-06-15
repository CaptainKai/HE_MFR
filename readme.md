# HE_MFR

The work of "[*Hypersphere guided Embedding for Masked Face Recognition*]".

## Introduction
Hypersphere Guided Embedding for Masked Face Recognition has been proposed to address the problem encountered in the Masked Face Recognition task, which arises due to non-biological information from occlusions. While some existing algorithms prefer to digesting the existence of masks by probing and covering, others aim to integrate face recognition and masked face recognition tasks into a unified solution domain. In this paper, we propose a framework to enable existing methods to accommodate multiple data distributions by orthogonal subspaces. Specifically, We introduce constraints on multiple hypersphere manifolds via Multi-Center Loss and employ a Spatial Split Strategy to ensure the orthogonality of base vectors associated with different hypersphere manifolds, corresponding to distinct distribution. Our method is extensively evaluated on publicly
available datasets on face recognition, mask face recognition and occlusion, demonstrating promising performance.

## Environment
The work is with **Python 3.7** and **PyTorch 1.7**.

## Pretrained Models

We provide 3 pretrained models:

| models | mixed |
| :-----:| :----: |
| [AMsoft36](https://pan.baidu.com/s/1D_bbmqalshR09dWFuc6N4g?pwd=xsd3) | ✘ |
| [ArcFace](https://pan.baidu.com/s/11s9cHa_WjUsOKDmHBUB_-w?pwd=r52s) | ✘ |
| [ours](https://pan.baidu.com/s/1oiJfHELEVfKR18s-HE23Ww?pwd=roxu) | ✔ |

Our model is trained on augmented MSIM-v1c dataset.


## Quick start
Type the following commands to train the model:
```
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train_v04.py
```
Type the following commands to extract features:
```
python3 test.py
```
