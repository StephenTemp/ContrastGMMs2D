''' 
utils.py : data utility functions to reduce clutter
'''

# IMPORTS
# ------------------------------------------
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import os
import glob
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------

# func: train_ind() => []
# return indices matching non-novel classes
def train_ind(dataset, CLASSES):
    indices =  []
    for i in range(len(dataset.data)):
        if dataset.data[i][1] in CLASSES:
            indices.append(i)

    return indices


# func: test_ind() => []
# return indices matching non-novel classes 
#                         + novel classes
def test_ind(dataset, CLASSES, PERC_VAL=0.20, val=False):
    indices =  []
    for i in range(len(dataset.data)):
        if dataset.data[i][0][1] in CLASSES:
            indices.append(i)
    
    # if this is the validation set, return PERC_VAL% of the data      
    if val == True: return indices[:int(PERC_VAL * len(indices))]
    else: return indices[int(PERC_VAL * len(indices)):]


def get_MNIST(KKC, ALL, BATCH_SIZE=64, VAL_PERC=0.20):

    trainset = torchvision.datasets.MNIST(root='./data', download=True, 
                                     transform=torchvision.transforms.ToTensor())

    train_inds = train_ind(trainset, CLASSES=KKC)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler = SubsetRandomSampler(train_inds))


    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, 
                                        transform=torchvision.transforms.ToTensor())
    val_inds = test_ind(testset, CLASSES=ALL, val=True, PERC_VAL=VAL_PERC)
    val_loader = DataLoader(testset, batch_size=BATCH_SIZE, sampler = SubsetRandomSampler(val_inds))

    test_inds = test_ind(testset, CLASSES=ALL, PERC_VAL=VAL_PERC)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE,
                                            sampler = SubsetRandomSampler(test_inds))

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return { "TRAIN" : train_loader, "VAL" : val_loader, "TEST" : test_loader, "CLASSES" : classes }


def showImg2d(dataloader):
    # FUNC: imshow( [] ) => None
    # SUMMARY: visualize the sampels
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    print(images[0].shape)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    labels = np.array(labels)
    df = pd.DataFrame(labels.reshape( (int(np.sqrt(len(images))), int(np.sqrt(len(images))) )))
    print(df)

'''
--------------------------------------------------------------------------------
ModelNet Functions
--------------------------------------------------------------------------------
'''
class ModelNet(Dataset):
    def __init__(self, ALL, train=True):
        self.train = train
        self.imgs_path = "./data/modelnet40v2png/"
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        
        self.data = []
        for i, class_path in enumerate(file_list):
            dir = None
            if train: dir = "/train"
            else: dir = "/test"
            class_name = class_path.split("/")[-1]

            cur_data = []
            for img_path in glob.glob(class_path + dir + "/*.png"):
                cur_data.append([img_path, class_name])
            
            if train: cur_data = np.array(cur_data)
            if not train: 
                cur_data.sort(key=lambda elem: elem[0])
                cur_data = np.array(np.split(np.array(cur_data), len(cur_data) / 80))
            
            self.data.extend(cur_data)

        cur_label = 0
        self.class_map = {}
        for class_name in ALL:
            self.class_map[class_name] = cur_label
            cur_label += 1
        self.img_dim = (224, 224)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.train:
            img_path, class_name = self.data[idx]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_dim)
            class_id = self.class_map[class_name]
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor[:, :, None]
            img_tensor = img_tensor.permute(2, 0, 1)
            class_id = torch.tensor([class_id])
        else: 
            samples = self.data[idx]
            img_paths, class_names = samples[:, 0], samples[:, 1]
            imgs = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in img_paths]
            imgs = np.array([cv2.resize(img, self.img_dim) for img in imgs])
            class_id = self.class_map[class_names[0]]
            img_tensor = torch.from_numpy(imgs)
            img_tensor = img_tensor[:, :, None]
            img_tensor = img_tensor.permute(0, 2, 1, 3)
            class_id = torch.tensor([class_id])

        return img_tensor, class_id



def get_ModelNet(KKC, ALL, BATCH_SIZE=64, VAL_PERC = 0.50):
    dataset_train = ModelNet(ALL=ALL)
    dataset_test = ModelNet(ALL=ALL, train=False)    

    train_inds = train_ind(dataset_train, CLASSES=KKC)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler = SubsetRandomSampler(train_inds))

    val_inds = test_ind(dataset_test, CLASSES=ALL, val=True, PERC_VAL=VAL_PERC)
    val_loader = DataLoader(dataset_test, batch_size=1, sampler = SubsetRandomSampler(val_inds))

    test_inds = test_ind(dataset_test, CLASSES=ALL, PERC_VAL=VAL_PERC)
    test_loader = DataLoader(dataset_test, batch_size=1,
                                            sampler = SubsetRandomSampler(test_inds))

    classes = list(dataset_test.class_map.keys())
    return { "TRAIN" : train_loader, "VAL" : val_loader, "TEST" : test_loader, "CLASSES" : classes }

