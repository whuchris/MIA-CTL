import torch
import numpy as np
from skimage import transform
from skimage import feature
from sklearn import preprocessing
import random
import os
import shutil

# Class DataUpdater, used to update the statistic data such as loss, accuracy.
class DataUpdater(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Class Matrics, used to calculate top_k accuracy, sensitivity, specificity, etc.
class Matrics(object):
    def accuracy(self,output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Pytorch 1.7
                res.append(correct_k.mul_(100.0 / batch_size))

            return res

# Class ImageProcessor, used to normalize, get augmentation or the local bianry pattern(LBP).
class ImageProcessor(object):
    def __init__(self, normalization=True,augmentation=True,use_lbp=False):
        self.normalization = normalization
        self.augmentation = augmentation
        self.use_lbp = use_lbp

    def __normalization(self,imgData):
        imgData = np.transpose(imgData,(2,0,1))
        imgData[0] = preprocessing.scale(imgData[1])
        imgData[1] = preprocessing.scale(imgData[1])
        imgData[2] = preprocessing.scale(imgData[1])
        imgData = np.transpose(imgData,(1,2,0))
        return imgData

    def __augmentation(self,imgData,flip_left_right_pro=0.5,flip_up_buttom_pro=0.5):
        if(random.random()<flip_left_right_pro):
            imgData = np.transpose(imgData,(2,0,1))
            imgData[0] = np.flip(imgData[0],1)
            imgData[1] = np.flip(imgData[1],1)
            imgData[2] = np.flip(imgData[2],1)
            imgData = np.transpose(imgData,(1,2,0))
        if(random.random()<flip_up_buttom_pro):
            imgData = np.transpose(imgData,(2,0,1))
            imgData[0] = np.flip(imgData[0],0)
            imgData[1] = np.flip(imgData[1],0)
            imgData[2] = np.flip(imgData[2],0)
            imgData = np.transpose(imgData,(1,2,0))
        return imgData

    def __call__(self, img):
        if self.use_lbp:
            img = feature.local_binary_pattern(img,P=24,R=3,method='ror').astype(float)
            img = img / (img.max() - img.min())
        imgData = transform.resize(img,(224,224,3),mode='constant',anti_aliasing=False)
        if self.normalization and self.use_lbp == False:
            imgData = self.__normalization(imgData)
        pos_1 = imgData.copy()
        pos_2 = imgData.copy()
        if self.augmentation:
            pos_1 = self.__augmentation(imgData)
            pos_2 = self.__augmentation(imgData)
        pos_1 = np.transpose(pos_1,(2,0,1))
        pos_2 = np.transpose(pos_2,(2,0,1))
        return pos_1,pos_2

def save_checkpoint_ssl(state,train_mode,model_name,is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join("./checkpoint",train_mode,model_name,filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,os.path.join("./checkpoint",train_mode,model_name,'model_best.pth.tar'))

def save_checkpoint(state, train_mode,model_name,is_best, filename='checkpoint.pth.tar',folder_num=0):
    filename = os.path.join("./checkpoint",train_mode,model_name,str(folder_num)+"_"+filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,os.path.join("./checkpoint",train_mode,model_name,str(folder_num)+'_model_best.pth.tar'))
