import sys
import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from tqdm import tqdm

trainpath = 'dataset/train/'
testpath = 'dataset/test1/'
words = os.listdir(trainpath)   # return a list containing all files and directories in a specified directory
category_number = len(words)   # how many characters in trainset

img_size = (128, 128)   # set image size uniformly

def loadOneWord(order):
    path = trainpath + words[order] + '/'   # each directory seperately 
    files = os.listdir(path)   # retrun a list containing all images of certain character
    datas = []
    for file in files:   # for each image in certain character directory, get file name one by one
        file = path + file
        img = np.asarray(Image.open(file))
        img = cv2.resize(img, img_size)
        datas.append(img)   # convert images into arrays, resize and append them to datas
    datas = np.array(datas)   # convert datas to array
    labels = np.zeros([len(datas), len(words)], dtype=np.uint8)   
    # create a characters * images matrix
    # len(array) returns count of the upper dimension(layer). np.zeros returns a matrix consituted by 0. unit8 is unsigned integer in 8bit
    labels[:, order] = 1   # change all the order-th item in the second dimension to 1
    return datas, labels

def transData():
    num = len(words)
    datas = np.array([], dtype=np.uint8)   # empty array for datas
    datas.shape = -1, 128, 128
    labels = np.array([], dtype=np.uint8)   # empty array for labels
    labels.shape = -1, 100
    for k in tqdm(range(num)):
        data, label = loadOneWord(k)

        datas = np.append(datas, data, axis=0)
        labels = np.append(labels, label, axis=0)

    np.save('data.npy', datas)
    np.save('label.npy', labels)

class TrainSet(data.Dataset):
    def __init__(self, eval=False):
        datas = np.load('data.npy')
        labels = np.load('label.npy')
        index = np.arange(0, len(datas), 1, dtype=np.int)
        np.random.seed(123)
        np.random.shuffle(index)
        if eval:
            index = index[:int(len(datas) * 0.1)]
        else:
            index = index[int(len(datas) * 0.1):]
        self.data = datas[index]
        self.label = labels[index]
        np.random.seed()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), \
               torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

def loadtestdata():
    files = os.listdir(testpath)
    datas = []
    for file in tqdm(files):
        file = testpath + file
        img = np.asarray(Image.open(file))
        img = cv2.resize(img, img_size)
        datas.append(img)
    datas = np.array(datas)
    return datas

if __name__ == '__main__':
    transData()
    # datas = np.load('data.npy')
    # labels = np.load('label.npy')
    # index = np.arange(0, len(datas), 1, dtype=np.int)
    # print(datas.shape, labels.shape)