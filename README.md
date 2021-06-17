# MIA-CTL
This repository contains all our materials except the whole dataset of our paper. Note that the .gitkeep file is **only for** keeping the integrity of our directory structure.

## Overview
We proposed a novel self-supervised learning using CTL for cervical OCT image classification. we use CNNs which can be implemented by various architectures, such as VGG, standard ResNet family, etc.  
The picture below demonstrates an overview of the proposed CTL framwork. The pretext task part uses texture contrast learning to extract feature maps from the texture features of the input images, and the downstream part linear classification is used to fine-tune the network Conv network.  
<img src="https://github.com/whuchris/MIA-CTL/blob/main/figures/Figure_1%20Framework.png" width="800"/><br/>
## Experiment Results
Our results are shown in the table below. This table mainly demonstrates accuracy,recall (sensitivity), specificity and AUC of our proposed method and the baselines, also provided in our paper.  
<img src="https://github.com/whuchris/MIA-CTL/blob/main/figures/Result.png" width="800"/><br/>
Compared to four human experts, the performance of our model is better. And the picture below displays the confusion matrices for both the five-classes and binary classification task.  
<img src="https://github.com/whuchris/MIA-CTL/blob/main/figures/Figure_4%20cfu_human%26net.png" width="800"/><br/>
Our paper gives more information about the performance of our proposed method. We give a result for an external validation samples which denotes that great generalization of our model. And we also give a visualization result of our model which denotes a better interpretability of our model. See the figure below for an example.  
<img src="https://github.com/whuchris/MIA-CTL/blob/main/figures/Figure_6%20Cross_shape.png" width="800"/><br/>
## Data preperation
By default. The `./dataset` directory stores our dataset. But we are sorry that our dataset cannot be made public due to the confidentiality agreement. You can get more information about our dataset by reading [`this paper`](https://www.medrxiv.org/content/10.1101/2020.05.12.20098830v1.full.pdf).  
Stll, you can make your own dataset if possible through the following steps.  
* Put all you dataset samples in directory `./dataset`. Keep sure that ***the label and patient name(if exists) are able to be obtained from the sample file name.***
* Execute method *generate_datatxt()* from `create_list.py` to generate a `data.txt` file in `./data_folder` directory. Modify the code in `create_list.py` if necessary. Keep sure that each line in `data.txt` is like '**[file_directory&emsp;label&emsp;patient_name(if exists)]**', where the spaces represent tabs('\t') with a newline('\n') in the end.  
*Example*:  `./dataset/0_P1.png  0 P1`.  
* Find the best P and R for your own dataset when execute method *local_binary_pattern()(LBP)* if you want to use CTL to boost model performance.  
* Execute method *split_ssl_and_sl()* from `create_list.py` to generate a `self_supervised_list_folder.txt` file for self-supervised training and a `supervised_folder.txt` file for 10-fold cross validation experiment. Execute method *generate_folder()* from `create_list.py` to generate `10_folder.txt` files for fine-tuning or training from scratch.
## Module envirment
The `requirement.txt` file records all dependencies and versions our work needs. Make sure your python version is 3.6.12 and to install all  dependencies at once, run `pip install -r requirements.txt`.  
## Runing
The `train.py` file defines all parameters and mode for each experiment.  
* Run self-supervised learning based on ResNet101 with CTL by:  
`python train.py --model_name 'lbp_resnet101' --epoch 150 --train_mode 'self_supervised' --use_lbp --lr 1e-2 --weight_decay 1e-6 --init --pre_train`
* Run self-supervised learning based on ResNet101 with SimCLR by:  
`python train.py --model_name 'resnet101' --epoch 150 --train_mode 'self_supervised' --lr 1e-2 --weight_decay 1e-6 --init --pre_train`
* Run self-supervised learning based on ResNet50 with SimCLR by:  
`python train.py --model_name 'resnet50' --epoch 150 --train_mode 'self_supervised' --lr 1e-2 --weight_decay 1e-6 --init --pre_train`
* Run supervised learning based on EfficientNet-B7 by:  
`python train.py --model_name 'efficientb7' --epoch 40 --train_mode 'supervised' --lr 5e-3 --init`
* Run supervised learning based on ResNet101 by:  
`python train.py --model_name 'resnet101' --epoch 40 --train_mode 'supervised' --lr 5e-3 --init`  
* Run supervised learning based on ResNet50 by:  
`python train.py --model_name 'resnet50' --epoch 40 --train_mode 'supervised' --lr 5e-3 --init`
* Run supervised learning based on VGG19 by:  
`python train.py --model_name 'vgg19' --epoch 40 --train_mode 'supervised' --lr 5e-3 --init`
* Fine-tune ResNet101 with CTL by:  
`python train.py --model_name 'lbp_resnet101' --epoch 40 --train_mode 'fine_tune' --use_lbp --lr 5e-3 --pre_load`
* Fine-tune ResNet101 with SimCLR by:  
`python train.py --model_name 'resnet101' --epoch 40 --train_mode 'fine_tune' --use_lbp --lr 5e-3 --pre_load`
* Fine-tune ResNet50 with SimCLR by:  
`python train.py --model_name 'resnet50' --epoch 40 --train_mode 'fine_tune' --use_lbp --lr 5e-3 --pre_load`  
* Make sure you have prepared your data and install all dependencies before running your experiments.
* You can change the superparameters such as `lr(learning rate)`,`weight_decay`, etc and you can define and change the models based on CNNs if necessary.
## Repository description  
***checkpoint***: `stores checkpoints for different train_mode when training on epoch.`    
&emsp;***fine_tune***: `stores checkpoints for different model on fine-tuning after self-supervised training.`  
&emsp;&emsp;***lbp_resnet101***: `stores state of resnet-101 using CTL on fine-tuning.`   
&emsp;&emsp;***resnet101***: `stores state of resnet-101 using SimCLR on fine-tuning.`  
&emsp;&emsp;***resnet50***: `stores state of resnet-50 using SimCLR on fine-tuning.`  
&emsp;***self_supervised***: `stores checkpoints for different model on self-supervised learning.`  
&emsp;&emsp;***lbp_resnet101***: `stores state of resnet-101 using CTL.`   
&emsp;&emsp;***resnet101***: `stores state of resnet-101 using SimCLR.`  
&emsp;&emsp;***resnet50***: `stores state of resnet-50 using SimCLR.`  
&emsp;***supervised***: `stores checkpoints for different model on training from scratch.`  
&emsp;&emsp;***resnet101***: `stores state of resnet-101 training from scratch.`  
&emsp;&emsp;***resnet50***: `stores state of resnet-50 training from scratch.`  
&emsp;&emsp;***vgg19***: `stores state of vgg19 training from scratch.`  
&emsp;&emsp;***efficientb7***: `stores state of efficientb7 pretrained on ImageNet.`  
***data_folder***: `contains the training, test, and validation "data.txt" for traning, test and validation.`  
***dataset***: `contains the whole dataset samples.`  
***figures***: `stores all figures displayed in our paper.`  
***result***: `same as checkpoint directory. stores the training results in .txt files.`  
***create_list.py***: `create data list from ./dataset. split data into different .txt files.`  
***data_reader.py***: `data loader class.`  
***model.py***: `define model structure for vgg and resnet family.`  
***train.py***: `training step.`  
***util.py***:  `uitlization function and class such as ImageProcessor.`  
***requirements.txt***:  `record all dependencies and versions.` 
## Contact us
If you have any issues, contact us at email address chris@whu.edu.cn or leave a message under [`Issues`](https://github.com/whuchris/MIA-CTL/issues) module!
