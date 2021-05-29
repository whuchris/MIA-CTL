from torch.utils.data import DataLoader
import os
import time
import shutil
from data_reader import MyDataSet
import argparse
import model
import torch.nn as nn
import torch
from util import DataUpdater,Matrics,save_checkpoint,save_checkpoint_ssl
from torch.optim import Adam
from torch.optim import SGD

def worker(args,folder_num=0):

    #model configuration
    model_name = args.model_name
    feature_dim = args.feature_dim
    num_class = args.num_class

    #trianing configuration
    epoch = args.epoch
    start_epoch = args.start_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_mode = args.train_mode
    temperature = args.temperature
    shuffle = not args.not_shuffle
    use_gpu = not args.not_use_gpu
    use_lbp = args.use_lbp

    #optimizer
    lr = args.lr
    weigth_decay = args.weight_decay

    #model params
    init = args.init
    pre_train = args.pre_train
    pre_load = args.pre_load

    #resume
    resume = args.resume
    resume_file_name = args.resume_file
    resume_file = ""

    #fine-tune
    load_path = args.pre_train_path

    #assert:
    assert model_name in ['resnet101','resnet50','vgg19','lbp_resnet101'] ,"invalid model_name %s"%(model_name)
    assert train_mode in ['self_supervised','supervised','fine_tune'] ,"invalid train_mode %s"%(train_mode)
    if resume:  resume_file = os.path.join("checkpoint",train_mode,model_name,resume_file_name)

    val_set = None
    if train_mode == 'self_supervised':
        train_set = MyDataSet(mode='ssl',use_lbp=use_lbp)
        test_set = MyDataSet(mode='test',augmentation=False,use_lbp=use_lbp)
    else:
        train_set = MyDataSet(folder_num = folder_num,mode='train',use_lbp=use_lbp)
        val_set = MyDataSet(folder_num = folder_num,mode='val',augmentation=False,use_lbp=use_lbp)
        test_set = MyDataSet(mode='test',augmentation=False,use_lbp=use_lbp)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=shuffle)
    if val_set != None:
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=shuffle)
    else:
        val_loader = None
    net = None
    if train_mode == 'self_supervised':
        if model_name in ['resnet101', 'lbp_resnet101']:
            net = model.ResNet101(init=init,pre_train=pre_train,feature_dim=feature_dim)
        elif model_name == 'resnet50':
            net = model.ResNet50(init=init,pre_train=pre_train,feature_dim=feature_dim)
        optimizer = Adam(net.parameters(), lr=lr, weight_decay=weigth_decay)
    else:
        net = model.Linear(model_name=model_name,num_class=num_class,init=init,
                           pre_train=pre_train,pre_load=pre_load,pretrained_path=load_path)
        optimizer = SGD(net.parameters(), lr=lr,momentum=0.9,weight_decay=weigth_decay)

    CE_loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available() and use_gpu:
        net.cuda()
        CE_loss.cuda()
        net=nn.DataParallel(net,device_ids=[0,2])

    best_acc = 0.0
    continue_epoch = 0

    #resume from checkpoint
    if resume == True:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file_name))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    model_settings = {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'val_loader': val_loader,
        'net': net,
        'optimizer': optimizer,
        'train_mode': train_mode,
        'temperature': temperature,
        'model_name': model_name,
        'continue_epoch': continue_epoch,
        'CE_loss': CE_loss,
        'best_acc': best_acc
    }
    if train_mode == 'self_supervised':
        self_supervised(start_epoch,epoch,**model_settings)
    else:
        fine_tune_or_supervised(start_epoch,epoch,**model_settings)

def self_supervised(start_epoch, epoch, **model_settings):
    train_loader = model_settings['train_loader']
    test_loader = model_settings['test_loader']
    net = model_settings['net']
    optimizer = model_settings['optimizer']
    train_mode = model_settings['train_mode']
    temperature = model_settings['temperature']
    model_name = model_settings['model_name']
    continue_epoch = model_settings['continue_epoch']
    best_acc = model_settings['best_acc']
    for i in range(start_epoch, epoch):
        _ssl_train(train_loader=train_loader,net=net,optimizer=optimizer,train_mode = train_mode,
                  temperature=temperature,model_name=model_name,i=i)

        #evaluate model
        test_acc = _ssl_test(test_loader=test_loader,net=net,train_mode=train_mode,
                            temperature=temperature,model_name=model_name,i=i)

        is_best = test_acc > best_acc
        best_acc = max(test_acc,best_acc)
        if is_best or (i+1) % 10 == 0:
            save_checkpoint_ssl({
                'epoch': i + 1,
                'arch': model_name,
                'state_dict': net.state_dict(),
                'best_acc': best_acc,
                'continue_epoch':continue_epoch,
                'optimizer': optimizer.state_dict(),
            }, train_mode= train_mode, model_name=model_name,is_best=is_best)

def fine_tune_or_supervised(start_epoch, epoch, **model_settings):
    train_loader = model_settings['train_loader']
    val_loader = model_settings['val_loader']
    net = model_settings['net']
    optimizer = model_settings['optimizer']
    train_mode = model_settings['train_mode']
    model_name = model_settings['model_name']
    continue_epoch = model_settings['continue_epoch']
    CE_loss = model_settings['CE_loss']
    best_acc = model_settings['best_acc']
    for i in range(start_epoch, epoch):
        _train(train_loader=train_loader,net=net,optimizer=optimizer,criterion=CE_loss,
              train_mode = train_mode,model_name=model_name,folder_num=folder_num,i=i)
        #测试
        val_acc = _val(val_loader=val_loader,net=net,criterion=CE_loss,
                      train_mode=train_mode,model_name=model_name,folder_num=folder_num,i=i)

        is_best = val_acc > best_acc
        best_acc = max(val_acc,best_acc)
        if is_best or (i+1) % 10 == 0:
            save_checkpoint({
                'epoch': i + 1,
                'arch': model_name,
                'state_dict': net.state_dict(),
                'best_acc': best_acc,
                'continue_epoch':continue_epoch,
                'optimizer': optimizer.state_dict(),
            }, train_mode= train_mode, model_name=model_name,is_best=is_best,folder_num=folder_num)

def _ssl_train(train_loader,net,optimizer,train_mode,temperature,model_name,i):
    batch_time = DataUpdater()
    losses = DataUpdater()
    tm = time.time()
    net.train()
    length = len(train_loader)
    for j,data in enumerate(train_loader):
        pos_1, pos_2,labels,name = data
        batch_size = len(labels)
        #pos_1, pos_2 = pos_1.cuda(non_blocking=True).float(), pos_2.cuda(non_blocking=True).float()
        pos_1, pos_2 = pos_1.float(), pos_2.float()
        if not args.not_use_gpu and torch.cuda.is_available():
            pos_1, pos_2 = pos_1.cuda(non_blocking=True),pos_2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)
        #top1.update(metrics.accuracy(output=outputs,target=labels,topk=(1,))[0].item(),batch_size)
        batch_time.update(time.time() - tm)

        tm = time.time()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss: {loss.val:.4f} (Avg:{loss.avg:.4f})\t'
              'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
            i+1, j * len(labels), len(train_loader.dataset),100. * j / length,
            loss=losses,batch_time=batch_time))
    try:
        log = os.path.join("result",train_mode,model_name,"train_result.txt")
        with open(log, 'a') as file:
            file.write('Epoch: [{0}]\t'
                       'Loss: {loss.avg:.4f}\t'
                       'Time: {batch_time.avg:.3f}\n'.format(
                i+1, loss=losses,batch_time=batch_time))
            file.close()
    except Exception as err:
        print(err)

def _ssl_test(test_loader,net,train_mode,temperature,model_name,i):
    batch_time = DataUpdater()
    top1 = DataUpdater()
    tm = time.time()
    net.eval()
    length = len(test_loader)
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for j, data_origin in enumerate(test_loader):
            data, _, labels, name = data_origin
            #data, labels = data.cuda(non_blocking=True).float(), labels.cuda(non_blocking=True)
            data = data.float()
            if not args.not_use_gpu and torch.cuda.is_available():
                data, labels = data.cuda(non_blocking=True),labels.cuda(non_blocking=True)
            feature, out = net(data)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # loop test data to predict the label by weighted knn search
        feature_labels = torch.tensor(test_loader.dataset.labels,device=feature_bank.device)
        for j, data_origin in enumerate(test_loader):
            data, _, target, name = data_origin
            #data,target = data.cuda(non_blocking=True).float(),target.cuda(non_blocking=True)
            data = data.float()
            if not args.not_use_gpu and torch.cuda.is_available():
                data, target = data.cuda(non_blocking=True),target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num = data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=1, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * 1, 5, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, 5) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 = torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            top1.update(total_top1/total_num,total_num)
            batch_time.update(time.time() - tm)
            tm = time.time()
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Top1Acc: {top1.val:.4f} (Avg:{top1.avg:.4f})\t'
                  'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
                i+1, j * len(labels), len(test_loader.dataset),100. * j / length,
                top1=top1,batch_time=batch_time))

        try:
            log = os.path.join("result",train_mode,model_name,"test_result.txt")
            with open(log, 'a') as file:
                file.write('Epoch: [{0}]\t'
                           'Top1: {top1.avg:.4f}\t'
                           'Time: {batch_time.avg:.3f}\n'.format(
                    i+1, top1=top1,batch_time=batch_time))
                file.close()
        except Exception as err:
            print(err)
    return top1.avg

def _train(train_loader,net,optimizer,criterion,train_mode,model_name,folder_num,i):
    batch_time = DataUpdater()
    losses = DataUpdater()
    top1 = DataUpdater()
    metrics = Matrics()
    tm = time.time()
    net.train()
    for j,data in enumerate(train_loader):
        pos_1, _,labels,name = data
        batch_size = len(labels)
        #pos_1 = pos_1.cuda(non_blocking=True).float()
        pos_1 = pos_1.float()
        if not args.not_use_gpu and torch.cuda.is_available():
            pos_1, labels = pos_1.cuda(non_blocking=True),labels.cuda(non_blocking=True)
        outputs = net(pos_1)
        #labels = labels.cuda()
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)
        top1.update(metrics.accuracy(output=outputs,target=labels,topk=(1,))[0].item(),batch_size)
        batch_time.update(time.time() - tm)
        tm = time.time()

        print('Folder: {}\t'
              'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss: {loss.val:.4f} (Avg:{loss.avg:.4f})\t'
              'AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\t'
              'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
            folder_num,i+1, j * len(labels), len(train_loader.dataset),100. * j / len(train_loader),
            loss=losses,top1=top1,batch_time=batch_time))
    try:
        log = os.path.join("result",train_mode,model_name,"train_result_"+str(folder_num)+".txt")
        with open(log, 'a') as file:
            file.write('Epoch: [{0}]\t'
                       'Loss: {loss.avg:.4f}\t'
                       'AccTop1: {top1.avg:.3f}\t'
                       'Time: {batch_time.avg:.3f}\n'.format(
                i+1, loss=losses,top1=top1,batch_time=batch_time))
            file.close()
    except Exception as err:
        print(err)

def _val(val_loader,net,criterion,train_mode,model_name,folder_num, i):
    batch_time = DataUpdater()
    losses = DataUpdater()
    top1 = DataUpdater()
    net.eval()
    metrics = Matrics()
    with torch.no_grad():
        tm = time.time()
        for j,data in enumerate(val_loader):
            imgs,_,labels,name = data
            batch_size = len(labels)
            # imgs = imgs.cuda().float()
            # labels = labels.cuda()
            imgs = imgs.float()
            if not args.not_use_gpu and torch.cuda.is_available():
                imgs, labels = imgs.cuda(non_blocking=True),labels.cuda(non_blocking=True)
            outputs = net(imgs)
            loss = criterion(outputs, labels)

            acc1 = metrics.accuracy(output=outputs, target=labels, topk=(1,))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - tm)
            tm = time.time()

            print('Folder: {}\t'
                  'Validation Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {loss.val:.4f}(Avg:{loss.avg:.4f})\t'
                  'AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\t'
                  'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
                folder_num,i+1, j * len(imgs), len(val_loader.dataset),100. * j / len(val_loader),
                loss=losses,top1=top1,batch_time=batch_time))

        print(' * AccTop1 {top1.avg:.3f}'
              .format(top1=top1))

        try:
            log = os.path.join("result",train_mode,model_name,"val_result_"+str(folder_num)+".txt")
            with open(log, 'a') as file:
                file.write('Loss {loss.avg:.4f} * AccTop1 {top1.avg:.3f}\n'.format(
                    loss=losses, top1=top1))
                file.close()
        except Exception as err:
            print(err)

    return top1.avg

if __name__ == '__main__':
    arg_setting = argparse.ArgumentParser()

    #model configuration
    arg_setting.add_argument('--model_name',default=r"resnet101",help="model name",type=str)
    arg_setting.add_argument('--feature_dim',default=128,help="size of feature",type=str)
    arg_setting.add_argument('--num_class',default=5,help="class num",type=str)

    #trianing configuration
    arg_setting.add_argument('--epoch',default=150,help="trainning epoch",type=int)
    arg_setting.add_argument('--start_epoch',default=0,help="start epoch(use for resume)",type=int)
    arg_setting.add_argument('--batch_size',default=2,help="tranning batch size",type=int)
    arg_setting.add_argument('--num_workers',default=0,help="data loader thread",type=int)
    arg_setting.add_argument('--train_mode',default="self_supervised",help="training mode",type=str)
    arg_setting.add_argument('--temperature', default=0.5, type=float, help='Temperature used in constrasive loss')
    arg_setting.add_argument('--not_shuffle',action='store_true',help='if shuffle the dataset')
    arg_setting.add_argument('--not_use_gpu',action='store_true',help='if use GPU to train')
    arg_setting.add_argument('--use_lbp', action='store_true', help='use lbp in self-supervised learning')

    #optimizer
    arg_setting.add_argument('--lr',default=1e-2,help="learning rate",type=float)
    arg_setting.add_argument('--weight_decay',default=0.0,help="learning rate",type=float)

    #model params
    arg_setting.add_argument('--init',action='store_true',help="random initialize model params")
    arg_setting.add_argument('--pre_train',action='store_true',help="train with self-supervised")
    arg_setting.add_argument('--pre_load',action='store_true',help="load the pretraining model learned by TCL or SimCLR")

    #resume
    arg_setting.add_argument('--resume',action='store_true',help="if need to resume from the latest checkpoint")
    arg_setting.add_argument('--resume_file',default='checkpoint.pth.tar',help="path to checkpoint",type=str)

    #fine-tune
    arg_setting.add_argument('--pre_train_path',default='checkpoint.pth.tar',help="file path for checkpoint",type=str)
    args = arg_setting.parse_args()
    folders = []
    if args.train_mode == 'self_supervised':
        folders = [0]
    else:
        folders = [0,1,2,3,4,5,6,7,8,9]
    for folder_num in folders:
        worker(args,folder_num = folder_num)