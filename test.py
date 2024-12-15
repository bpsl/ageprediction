import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, sys
import joblib
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#cudnn.benchmark = True

from dataloadallinone import LungAgeData

#from utils.mape import MAPELoss



splitratio = 0.1
gender = 'male'
seed = 1


model = torch.load('mode.pt',map_location='cuda:0')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
if isinstance(model, nn.DataParallel): 
    model = model.module



#overall test, and draw data for scatter plot and spearman test
val_dataset = LungAgeData(split='test', gender=gender, sourcedata='external',  sourcedata='Tenthdataset', maxage=0,minage=5760)
val_dataloader = DataLoader(val_dataset, batch_size=128, drop_last = False, shuffle=False)



predictedvaluelist = []
truevaluelist = []
model.eval()
maelist, mselist, rmselist, mapelist = [], [], [], []
for images, labels, filenames in val_dataloader: 
    images=images.cuda()
    labels=labels.cuda()
    outputs = model(images).squeeze()
    mae = torch.mean(torch.abs(outputs - labels))
    mape = torch.mean(torch.abs((labels - outputs) / (labels + 1e-8))) 
    maelist.append(mae.item())
    mapelist.append(mape.item())
    
    outputslist = outputs.tolist()
    labelslist = labels.tolist()
    filenameslist = list(filenames)
    
    predictedvaluelist.extend(outputslist)
    truevaluelist.extend(labelslist)

nparraymaelistmean = np.array(maelist).mean()
nparraymapelistmean = np.array(mapelist).mean()
print(nparraymaelistmean, nparraymapelistmean)



###############################################################################
def evaluate(mod, loader): #model, dataloader
    mod.eval()
    maelist, mapelist = [], []
    for images, labels, filenames in loader: 
        images=images.cuda()
        labels=labels.cuda()
        outputs = mod(images).squeeze()
        #print('labels', labels.int())
        #print('outputs', outputs.int())
        mae = torch.mean(torch.abs(outputs - labels))
        mape = torch.mean(torch.abs((labels - outputs) / (labels + 1e-8))) #Mean Absolute Percentage Error
        maelist.append(mae.item())
        mapelist.append(mape.item())       
    nparraymaelistmean = np.array(maelist).mean()
    nparraymapelistmean = np.array(mapelist).mean()
    print(nparraymaelistmean, nparraymapelistmean) 
    return nparraymaelistmean
###############################################################################


#evaluation of each age group
test_datasetlist3 = []
test_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='external', savefilename='Tenthdataset', minage=0,maxage=359)) 
test_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='external', savefilename='Tenthdataset', minage=360,maxage=1079))
test_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='external', savefilename='Tenthdataset', minage=1080,maxage=2159))
test_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='external', savefilename='Tenthdataset', minage=2160,maxage=3959))
test_datasetlist3.append(LungAgeData(split='test', gender=gender, sourcedata='external', savefilename='Tenthdataset', minage=3960,maxage=5760)) 
test_dataloderlist3 = []
for k in range(len(test_datasetlist3)):
    test_dataloderlist3.append(DataLoader(test_datasetlist3[k], batch_size=32, drop_last = False, shuffle=False))


for test_dataloaderj in test_dataloderlist3:
    evaluate(model, test_dataloaderj)









###############################################################################
#Plot scatter plots

plt.figure(figsize=(10, 10))
plt.scatter(truevaluelist, predictedvaluelist, alpha=0.5, s=4)
plt.title('External validation set, male')
plt.xlabel('True age (month)')
plt.ylabel('Predicted age (month)')

#plt.axis('equal') 
plt.xlim(0, 210) 
plt.ylim(0, 210)
plt.plot([0, 210], [0, 210], 'r--', lw=2) 

#plt.grid(True)
#plt.legend()# 
plt.show()
###############################################################################





###############################################################################
#Spearman test
from scipy import stats
spearman_corr, p_value = stats.spearmanr(truevaluelist, predictedvaluelist)

print(f"Spearman Correlation Coefficient: {spearman_corr}")
print(f"P-value: {p_value}")
###############################################################################









