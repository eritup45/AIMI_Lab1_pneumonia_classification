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

# VISUALIZE
import wandb
wandb.init(project="NYCU_AIMI_LAB1", entity="eritup45")
wandb.config = {
    "IMAGE_SIZE":(128,128), # (256, 256),  
    "batch_size":128, 
    "learning_rate": 0.1, # 0.01
    "epochs": 30,
}

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

def imshow(img):
    plt.figure(figsize=(20, 20))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def Convlayer(in_channels,out_channels,kernel_size,padding=1,stride=1):
    conv =  nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    return conv

class NeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(NeuralNet,self).__init__()
        
        self.conv1 = Convlayer(in_channels=3,out_channels=32,kernel_size=3)
        self.conv2 = Convlayer(in_channels=32,out_channels=64,kernel_size=3)
        self.conv3 = Convlayer(in_channels=64,out_channels=128,kernel_size=3)
        self.conv4 = Convlayer(in_channels=128,out_channels=256,kernel_size=3)
        self.conv5 = Convlayer(in_channels=256,out_channels=512,kernel_size=3)
        
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=x.view(-1,512*4*4)
        x=self.classifier(x)

        return x

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

def resnet18(num_class):
    resnet18 = torchvision.models.resnet18()
    resnet18.fc.out_features = num_class
    return resnet18

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

def training(model, train_loader, test_loader, Loss, optimizer, epochs, device, num_class, name):
    model.to(device)
    best_model_wts = None
    best_evaluated_acc = 0
    train_acc = []
    test_acc = []
    test_Recall = []
    test_Precision = []
    test_F1_score = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.96)
    for epoch in range(1, epochs+1):
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for idx,(data, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                        
                data = data.to(device,dtype=torch.float)
                label = label.to(device,dtype=torch.long)

                predict = model(data)      

                loss = Loss(predict, label.squeeze())

                total_loss += loss.item()
                pred = torch.max(predict,1).indices
                correct += pred.eq(label).cpu().sum().item()
                        
                loss.backward()
                optimizer.step()

            total_loss /= len(train_loader.dataset)
            correct = (correct/len(train_loader.dataset))*100.
            print ("Epoch : " , epoch)
            print ("Loss : " , total_loss)
            print("LR: ", optimizer.param_groups[0]['lr'])
            print ("Correct : " , correct)
            #print(epoch, total_loss, correct) 
                
        scheduler.step()
        accuracy  , Recall , Precision , F1_score = evaluate(model, device, test_loader)
        train_acc.append(correct)  
        test_acc.append(accuracy)
        test_Recall.append(Recall)
        test_Precision.append(Precision)
        test_F1_score.append(F1_score)

        wandb.log({
            'train_epoch': epoch, 
            'train_loss': loss,
            'train_acc': correct,
            'train_LR': optimizer.param_groups[0]['lr'],
            'test_acc': accuracy,
            'test_Recall': Recall,
            'test_Precision': Precision, 
            'test_F1_score': F1_score
            }) 

        if accuracy > best_evaluated_acc:
            best_evaluated_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # save best model
        torch.save(best_model_wts, f"{name}_{best_evaluated_acc: .2f}.pt")

        # Save the model checkpoint. This automatically saves a file to the cloud
        wandb.save(f'{name}_{best_evaluated_acc: .2f}.pt')

        model.load_state_dict(best_model_wts)

    return train_acc , test_acc , test_Recall , test_Precision , test_F1_score

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

    best_model_wts = None
    train_path='archive/chest_xray/train'
    test_path='archive/chest_xray/test'
    val_path='archive/chest_xray/val'

    trainset=datasets.ImageFolder(train_path,transform=images_transforms('train'))
    testset=datasets.ImageFolder(test_path,transform=images_transforms('test'))
    valset=datasets.ImageFolder(val_path,transform=images_transforms('val'))

    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader = DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=2)

    examples=iter(train_loader)
    images,labels=examples.next()
    print("images.shape: ", images.shape)
    # imshow(torchvision.utils.make_grid(images[:56],pad_value=20))

    # display classes -> index
    class_names = trainset.classes
    print("trainset class index convertion: ", trainset.class_to_idx)
    
    # model = denseNet(2)
    model = resnext50_32x4d(2)
    # model = resnet18(2)
    # model = efficientnet_b3(2)
    # model = ResNet50(2, True)
    # model = NeuralNet(2)
    if(best_model_wts):
        model.load_state_dict(torch.load(best_model_wts))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # print (summary(model,(3,128,128)))

    print (train_loader)
    dataiter = iter(train_loader)
    images , labels = dataiter.next()
    print (type(images), type(labels))
    print (images.size(), labels.size())

    # wandb.watch() automatically fetches all layer dimensions, gradients, model parameters
    # and logs them automatically to your dashboard.
    wandb.watch(model, log="all")
    train_acc , test_acc , test_Recall , test_Precision , test_F1_score  = training(model, train_loader, test_loader, criterion, optimizer,30, device, 2, 'CNN_chest')


