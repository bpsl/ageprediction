import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, sys
import joblib
import random
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import copy

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
#cudnn.benchmark = True



splitratio = 0.1
gender = 'male'
seed = 1
loadcheckpointname = ''
num_epochs = 1000
initial_lr = 0.0001
###############################################################################
###############################################################################


if __name__ == '__main__':
    
    from dataloadallinone import LungAgeData
    from dataloadallinone import buildinternaldataset, buildexternaldataset

    #Training set
    train_dataset = LungAgeData(split='train', gender=gender, sourcedata='internal', splitratio=splitratio, savefilename='Xinhuadataset', splitseed=seed, maxage=5760,minage=0)
    train_dataloader = DataLoader(train_dataset, batch_size=128, drop_last = False, shuffle=True)
    
    #Internal validation set overall
    val_dataset = LungAgeData(split='test', gender=gender, sourcedata='internal', splitratio=splitratio, savefilename='Xinhuadataset', splitseed=seed, maxage=5760,minage=0)
    val_dataloader = DataLoader(val_dataset, batch_size=256, drop_last = False, shuffle=False)

    #Internal validation set each age group
    val_datasetlist3 = []
    val_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='internal', splitratio=splitratio, datafilename='Xinhuadataset', splitseed=seed, minage=0,maxage=359)) 
    val_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='internal', splitratio=splitratio, datafilename='Xinhuadataset', splitseed=seed, minage=360,maxage=1079)) 
    val_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='internal', splitratio=splitratio, datafilename='Xinhuadataset', splitseed=seed, minage=1080,maxage=2159)) 
    val_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='internal', splitratio=splitratio, datafilename='Xinhuadataset', splitseed=seed, minage=2160,maxage=3959)) 
    val_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='internal', splitratio=splitratio, datafilename='Xinhuadataset', splitseed=seed, minage=3960,maxage=5760)) 
    val_dataloderlist3 = []
    for k in range(len(val_datasetlist3)):
        val_dataloderlist3.append(DataLoader(val_datasetlist3[k], batch_size=32, drop_last = False, shuffle=False))


    #Load Trained
    if loadcheckpointname != '':
        model = torch.load(loadcheckpointname,map_location='cuda:0')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        if isinstance(model, nn.DataParallel):
            model = model.module
    #Train from scratch
    else:
        from models.resnetcoordatt import resnet18_coordatt
        model = resnet18_coordatt()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    model = model.cuda()
    if num_gpus >= 2: 
        model = nn.DataParallel(model, device_ids=[0, 1]) 

    #Loss
    criterion = nn.L1Loss()
    
    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    def evaluate(mod, loader): #model, dataloader
        mod.eval()
        maelist, mapelist = [], []
        for images, labels, filenames in loader: 
            images=images.cuda()
            labels=labels.cuda()
            outputs = mod(images).squeeze()
            mae = torch.mean(torch.abs(outputs - labels))
            mape = torch.mean(torch.abs((labels - outputs) / (labels + 1e-8))) #Mean Absolute Percentage Error
            maelist.append(mae.item())
            mapelist.append(mape.item())       
        nparraymaelistmean = np.array(maelist).mean()
        nparraymapelistmean = np.array(mapelist).mean()
        print(epoch+1, nparraymaelistmean, nparraymapelistmean) 
        return nparraymaelistmean
        
    ###############################################################################
    ###############################################################################
    for epoch in range(num_epochs):
        print('epoch', epoch)
        
        #Training
        model.train()
       
        for images, labels, filenames in train_dataloader:
            images=images.cuda()
            labels=labels.cuda()
            optimizer.zero_grad()
            outputs = model(images).squeeze()  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #print(loss)
            try:
                scheduler.step() 
            except:
                pass
            
        #Validation
        #Internal validation set overall 
        mae = evaluate(model, val_dataloader)
        #Internal validation set each age gourp
        for val_dataloaderi in val_dataloderlist3:
            evaluate(model, val_dataloaderi)
     
        #save model
        torch.save(model, str(epoch)+'.pt')
    
    #from torchsummary import summary #Conclude model if needed
    #summary(model, input_size=(1, 256, 256))
    
    
    


