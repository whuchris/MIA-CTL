from torch.utils.data import Dataset
import os
from skimage import io
from util import ImageProcessor

class MyDataSet(Dataset):
    def __init__(self, folder_num = 0,mode='train',normalization=True,augmentation=True,use_lbp=False):
        assert (mode in ['ssl','train','test','val']),"mode must be type in ['ssl','train','test','val']"
        fileNames = self.__load_file(mode=mode,folder_num=folder_num)
        self.imgs=[]
        self.labels=[]
        self.img_processor = ImageProcessor(normalization=normalization,augmentation=augmentation,use_lbp=use_lbp)
        for fileName in fileNames:
            self.imgs.append(fileName.split('\t')[0])
            self.labels.append(int(fileName.split('\t')[1]))

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        label = self.labels[index]
        img = io.imread(imgPath)
        pos_1, pos_2 = self.img_processor(img)
        return [pos_1,pos_2,label,imgPath]

    def __len__(self):
        return len(self.imgs)


    def __load_file(self,pre = "data_folder",mode='train',folder_num = 0):
        if mode == 'test':
            mode+= '_folder.txt'
        elif mode == 'ssl':
            mode = 'self_supervised_list_folder.txt'
        else:
            mode+= '_folder_' +str(folder_num)+'.txt'
        mode = os.path.join(pre,mode)
        fileNames=[]
        with open(mode,'r') as file:
            lines = file.readlines()
            for line in lines:
                fileNames.append(line.strip(("\n")))
        return fileNames



