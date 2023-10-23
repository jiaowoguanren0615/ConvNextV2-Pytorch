# ConvNextV2-Pytorch

## Paper: ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders(https://arxiv.org/pdf/2301.00808.pdf)
## Precautions
The code warehouse mainly contains two models: ConvNextV2 and FCMAE-ConvNextV2. The code has been slightly modified based on the source code of the facebookresearch warehouse. You can use your own data set to train the classification model and FCMAE model.

Note: Before training the classification model, you need to enter the ___train_gpu_finetune.py___ file to modify the path and batchsize of your own data set. If it is a single-gpu environment, you need to replace the "--sync-bn" parameter with false, otherwise it may An error will be reported; if you want to train the FCMAE model, you need to first enter the "util-->__init__" file, modify the import package path, then enter ___train_gpu_pretrain.py___, modify the path of the data set, and finally run the ___train_gpu_pretrain.py___ file.

## TRAIN AND EVALUATE CLF MODEL
1. cd ConvNextV2
2. python train_gpu_finetune.py
