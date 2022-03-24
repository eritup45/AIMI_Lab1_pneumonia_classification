import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
from torchvision import datasets
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from torchvision import models
# from tqdm import tqdm_notebook as tqdm
import time
from tqdm import tqdm
import warnings
import copy
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from torchsummary import summary
from sklearn.metrics import accuracy_score,classification_report, f1_score,roc_auc_score


def images_transforms(phase):
    if phase == 'training':
        data_transformation =transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomEqualize(10),
            transforms.RandomRotation(degrees=(-25,20)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    return data_transformation


class ResNet50(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet50,self).__init__()
        self.model=models.resnet50(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        
   def forward(self,X):
        out=self.model(X)
        return out

def denseNet(num_class):
    densenet = models.densenet161(pretrained=True)
    densenet.classifier.out_features = num_class
    return densenet


def resnext50_32x4d(num_class):
    resnext50_32x4d = torchvision.models.resnext50_32x4d(pretrained=True)
    resnext50_32x4d.fc.out_features = num_class
    return resnext50_32x4d

def efficientnet_b3(num_class):
    from efficientnet_pytorch import EfficientNet
    from torch import nn
    model = EfficientNet.from_pretrained('efficientnet-b3')
    feature = model._fc.in_features
    model._fc = nn.Linear(in_features=feature,out_features=num_class,bias=True)
    return model


def evaluate(model, device, test_loader):
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1
        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)

        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        Recall = TP/(TP+FN)
        print ("Recall : " ,  Recall )

        Precision = TP/(TP+FP)
        print ("Preecision : " ,  Precision )

        F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : " , F1_score)

        correct = (correct/len(test_loader.dataset))*100.
        print ("Accuracy : " , correct ,"%")

    return correct , Recall , Precision , F1_score

if __name__=="__main__":
    IMAGE_SIZE=(128,128) # (256, 256) 
    batch_size=128
    learning_rate = 0.001 # 0.01
    epochs=30
    num_classes=2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)

    best_model_wts = "./resnext50_img128_adam0.001_ep24_acc94.39_BEST.pt"
    # train_path='archive/chest_xray/train'
    test_path='archive/chest_xray/test'
    val_path='archive/chest_xray/val'

    testset=datasets.ImageFolder(test_path,transform=images_transforms('test'))
    valset=datasets.ImageFolder(val_path,transform=images_transforms('val'))

    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader = DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=2)

    # model = denseNet(2)
    model = resnext50_32x4d(2)
    if(best_model_wts):
        model.load_state_dict(torch.load(best_model_wts))
    model.to(device)

    accuracy  , Recall , Precision , F1_score = evaluate(model, device, test_loader)

