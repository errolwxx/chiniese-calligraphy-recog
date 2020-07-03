import pandas as pd
import os
import torch
# import torch.optim as optim
# import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import model as model
import data
import cv2
from PIL import Image

trainpath = './dataset/train/'
testpath = './dataset/test/'
test_label = os.listdir(testpath)
test = []
labels = []
words = os.listdir(trainpath)
words = np.array(words)
img_size = (128, 128)

net = model.net()
if torch.cuda.is_available():
    net.cuda()
net.eval()


if __name__ == '__main__':
    checkpoint = model.load_checkpoint()
    net.load_state_dict(checkpoint['state_dict'])

    for labelDir in test_label:
        data_by_label = os.listdir(testpath + labelDir)
        for item in data_by_label:
            datapath = testpath + labelDir + "/" + item
            label = os.path.join(datapath, os.pardir)
            label = os.path.relpath(label)
            label = os.path.basename(label)
            labels.append(label)
        test.extend(data_by_label)

    predicts = []
    for lb in test_label:
        cwd = testpath + lb + "/"
        cwd_items = os.listdir(cwd)
        # print(cwd_items)
        for item in cwd_items:

            img = np.asarray(Image.open(cwd + item).convert('L'))
            img = cv2.resize(img, img_size)
            img = np.array(img)
            img.astype(np.float)

            pred_choice = []
            img = torch.from_numpy(img)
            char = Variable(img).cuda().float()
            char = char.view(-1, 1, 128, 128)

            output = net(char)
            output = output.cpu()
            output = output.data.numpy()

            index = np.argmax(output)
            pred_choice.append(index)
            pre = np.array(pred_choice)
            label_index = pre[0]
            predict = words[label_index]
            predict = list(predict.flatten())

            # print(i, predict)
            predicts.append(predict)
            predicts = [item for sublist in predicts for item in sublist]
            
    
    trues = []
    falses = []
    checks = []
    for predict_num in range(len(predicts)):
        check = (predicts[predict_num]==labels[predict_num])
        checks.append(str(check))

    acc = checks.count('True')/len(checks)
    dataframe = pd.DataFrame({'filename': test, 'label': labels, 'predict_label': predicts, 'check': checks})
    dataframe.to_csv("test.csv", index=False, encoding='utf-8')
    read = pd.read_csv('test.csv')
    print(read, acc)