# MIA-TCL
This repository contains all our materials except the whole dataset of our paper. Note that the .gitkeep file is **only for** keeping the integrity of our directory structure.

## Overview
We proposed a novel self-supervised learning using TCL for cervical OCT image classification. we use CNNs which can be implemented by various architectures, such as VGG, standard ResNet family, etc.  
The picture below demonstrates an overview of the proposed TCL framwork. The pretext task part uses texture contrast learning to extract feature maps from the texture features of the input images, and the downstream part linear classification is used to fine-tune the network Conv network.

![image](https://github.com/ChrisNieo/MIA-TCL/blob/main/figures/Figure_2%20Framework.png)

## Repository description
### Description of each directory and file.  
**checkpoint**:                stores checkpoints for different train_mode when training on epoch.    
&emsp;**fine_tune**:           stores checkpoints for different model on fine-tuning after self-supervised training.  
&emsp;&emsp;**lbp_resnet101**: stores state of resnet-101 using TCL on fine-tuning.   
&emsp;&emsp;**resnet101**:     stores state of resnet-101 using SimCLR on fine-tuning.  
&emsp;&emsp;**resnet50**:      stores state of resnet-50 using SimCLR on fine-tuning.  
&emsp;**self_supervised**:     stores checkpoints for different model on self-supervised learning.  
&emsp;&emsp;**lbp_resnet101**: stores state of resnet-101 using TCL.   
&emsp;&emsp;**resnet101**:     stores state of resnet-101 using SimCLR.  
&emsp;&emsp;**resnet50**:      stores state of resnet-50 using SimCLR.  
&emsp;**supervised**:          stores checkpoints for different model on training from scratch.  
&emsp;&emsp;**resnet101**:     stores state of resnet-101 training from scratch.   
&emsp;&emsp;**resnet50**:      stores state of resnet-50 training from scratch.  
&emsp;&emsp;**vgg19**:         stores state of vgg19 training from scratch.  
data_folder  
dataset  
figures  
result  
&emsp;fine_tune  
&emsp;&emsp;lbp_resnet101  
&emsp;&emsp;resnet101  
&emsp;&emsp;resnet50  
&emsp;self_supervised  
&emsp;&emsp;lbp_resnet101  
&emsp;&emsp;resnet101  
&emsp;&emsp;resnet50  
&emsp;supervised  
&emsp;&emsp;resnet101  
&emsp;&emsp;resnet50  
&emsp;&emsp;vgg19  
create_list.py  
data_reader.py  
model.py  
train.py  
util.py  
