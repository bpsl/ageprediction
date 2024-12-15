import joblib
import random     
import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split





def buildinternaldataset(savefilename, trainingsetlocation):

    filelist = os.listdir(trainingsetlocation)
    malefilelist =  [x for x in filelist if '_m_' in x]
    femalefilelist = [x for x in filelist if '_f_' in x] 
    
    male_unsqueezedarraylist = []
    for filename in malefilelist:
        imagepath = trainingsetlocation + '/' + filename
        img = Image.open(imagepath)
        
        resizedimg = img.resize((256,256))
        resizedarray = np.array(resizedimg)[:,:,1]
        unsqueezedarray = np.resize(resizedarray, [1,256,256])
        male_unsqueezedarraylist.append(unsqueezedarray)
  
    female_unsqueezedarraylist = []
    for filename in femalefilelist:
        imagepath = trainingsetlocation + '/' + filename
        img = Image.open(imagepath)
        
        resizedimg = img.resize((256,256))
        resizedarray = np.array(resizedimg)[:,:,1]
        unsqueezedarray = np.resize(resizedarray, [1,256,256])
        female_unsqueezedarraylist.append(unsqueezedarray)
        
    np.savez(savefilename+'male.npz', 
             filenames = np.array(malefilelist), 
             unsqueezedarrays = np.array(male_unsqueezedarraylist))
    np.savez(savefilename+'female.npz',            
             filenames = np.array(femalefilelist), 
             unsqueezedarrays = np.array(female_unsqueezedarraylist))





def buildexternaldataset(savefilename, trainingsetlocation, validationsetlocation):
    #if 数据sourcedata == 'external' and not os.path.exists(数据储存名+'maletrain.npz'):  
    trainfilelist = os.listdir(validationsetlocation)
    testfilelist = os.listdir(validationsetlocation)
    maletrainfilelist = [x for x in trainfilelist if '_m_' in x] 
    maletestfilelist = [x for x in testfilelist if '_m_' in x] 
    femaletrainfilelist = [x for x in trainfilelist if '_f_' in x] 
    femaletestfilelist = [x for x in testfilelist if '_f_' in x] 

    unsqueezedarraylist = []
    for filename in maletrainfilelist:
        imagepath = trainingsetlocation + '/' + filename
        img = Image.open(imagepath)
        
        resizedimg = img.resize((256,256))
        resizedarray = np.array(resizedimg)[:,:,1]
        unsqueezedarray = np.resize(resizedarray, [1,256,256])
        unsqueezedarraylist.append(unsqueezedarray)
    np.savez(savefilename+'maletrain.npz', filenames = np.array(maletrainfilelist), 
            unsqueezedarrays = np.array(unsqueezedarraylist))

    unsqueezedarraylist = []
    for filename in femaletrainfilelist:
        imagepath = trainingsetlocation + '/' + filename
        img = Image.open(imagepath)
        
        resizedimg = img.resize((256,256))
        resizedarray = np.array(resizedimg)[:,:,1]
        unsqueezedarray = np.resize(resizedarray, [1,256,256])
        unsqueezedarraylist.append(unsqueezedarray)
    np.savez(savefilename+'femaletrain.npz', filenames = np.array(femaletrainfilelist), 
            unsqueezedarrays = np.array(unsqueezedarraylist))

    unsqueezedarraylist = []
    for filename in maletestfilelist:
        imagepath = validationsetlocation + '/' + filename
        img = Image.open(imagepath)
        
        resizedimg = img.resize((256,256))
        resizedarray = np.array(resizedimg)[:,:,1]
        unsqueezedarray = np.resize(resizedarray, [1,256,256])
        unsqueezedarraylist.append(unsqueezedarray)
    np.savez(savefilename+'maletest.npz', filenames = np.array(maletestfilelist), 
            unsqueezedarrays = np.array(unsqueezedarraylist)) 

    unsqueezedarraylist = []
    for filename in femaletestfilelist:
        imagepath = validationsetlocation + '/' + filename
        img = Image.open(imagepath)
        
        resizedimg = img.resize((256,256))
        resizedarray = np.array(resizedimg)[:,:,1]
        unsqueezedarray = np.resize(resizedarray, [1,256,256])
        unsqueezedarraylist.append(unsqueezedarray)
    np.savez(savefilename+'femaletest.npz', filenames = np.array(femaletestfilelist), 
            unsqueezedarrays = np.array(unsqueezedarraylist)) 




class LungAgeData(Dataset):
    def __init__(self, split='train', gender='male', splitseed=1, sourcedata='internal', splitratio=0.1, savefilename='', minage=0, maxage=5760): 
    
        self.splitseed = splitseed 
        self.split = split
        self.gender = gender
            
        #internal
        if sourcedata == 'internal':
            
            if self.gender == 'male':
                self.data = np.load(savefilename+'male.npz')
            elif self.gender == 'female':
                self.data = np.load(savefilename+'female.npz') 
            
            filenames = self.data['filenames']
            unsqueezedarrays = self.data['unsqueezedarrays']

        
            del self.data 
            
            if self.split == 'train': 
                self.unsqueezedarrays2, _, self.filenames2, _ = \
                    train_test_split(unsqueezedarrays, filenames, test_size=splitratio, random_state=splitseed)
            elif self.split == 'test': 
                _, self.unsqueezedarrays2, _, self.filenames2 = \
                    train_test_split(unsqueezedarrays, filenames, test_size=splitratio, random_state=splitseed)   


        #external
        if sourcedata == 'external':
            if self.split == 'train':
                if self.gender == 'male':
                    alldata = np.load(savefilename+'maletrain.npz')
                elif self.gender == 'female':
                    alldata = np.load(savefilename+'femaletrain.npz')
            elif self.split == 'test':
                if self.gender == 'male':
                    alldata = np.load(savefilename+'maletest.npz')
                elif self.gender == 'female':
                    alldata = np.load(savefilename+'femaletest.npz')
            
            self.filenames2 = alldata['filenames'] 
            self.unsqueezedarrays2 = alldata['unsqueezedarrays'] 
            
            del alldata

        ###########################
        ###########################
        
        toberemovedlist = []
        for i in range(len(self.filenames2)):
            filename = self.filenames2[i]
            age = int(filename.split('！')[-1].split('.')[0].split('_')[-1])
            if age < minage or age > maxage:
                toberemovedlist.append(i)
        

        self.filenames = np.delete(self.filenames2, toberemovedlist)
        self.unsqueezedarrays = np.delete(self.unsqueezedarrays2, toberemovedlist, axis=0)

        del self.filenames2, self.unsqueezedarrays2

        
        
    def __len__(self):
        return len(self.unsqueezedarrays)
    
    
    
    def __getitem__(self, index): 

        filename = self.filenames[index]
        age = filename.split('！')[-1].split('.')[0].split('_')[-1]

        agemonth = torch.tensor(int(age)//30).float()

        unsqueezedarray = self.unsqueezedarrays[index]
        unsqueezedtensor = torch.from_numpy(unsqueezedarray).float()
        
        return unsqueezedtensor, agemonth, filename#, index













