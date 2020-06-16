from PIL import Image
from torchvision import transforms
from model import net
from torch.autograd import Variable as V
import torch as t
 
trans=transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
        ])
 
#读入图片
img = Image.open('Fu.jpg').convert('L')

img=trans(img)#这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
img = img.unsqueeze(0)#增加一维，输出的img格式为[1,C,H,W]

pred_model = net()
pred_model.cuda()#导入网络模型
pred_model.eval()

pred_model = t.nn.DataParallel(pred_model)
pred_model.load_state_dict(t.load('model_save/model_parameters.pth.tar'))#加载训练好的模型文件
 
input = V(img.cuda())
score = pred_model(input)#将图片输入网络得到输出
probability = t.nn.functional.softmax(score,dim=1)#计算softmax，即该图片属于各类的概率
max_value,index = t.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别
print(index)


# import torch
# import cv2
# import torch.nn.functional as F
# from model import net  ##重要，虽然显示灰色(即在次代码中没用到)，但若没有引入这个模型代码，加载模型时会找不到模型
# import torch
# from torch.autograd import Variable
# from torchvision import datasets, transforms
# import numpy as np
 
# def pred():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = torch.load('model_save/model_parameters.pth.tar') #加载模型
#     model = model.to(device)
#     model.eval()    #把模型转为test模式

#     img = cv2.imread("Fu.jpg")  #读取要预测的图片
#     trans = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#图片转为灰度图，因为mnist数据集都是灰度图
#     img = trans(img)
#     img = img.to(device)
#     img = img.unsqueeze(0)  #图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
#     #扩展后，为[1，1，28，28]
#     output = model(img)
#     prob = F.softmax(output, dim=1)
#     prob = Variable(prob)
#     prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
#     print(prob)  #prob是10个分类的概率
#     pred = np.argmax(prob) #选出概率最大的一个
#     print(pred.item())

# if __name__ =='__main__':
#     pred()

# import torch
# torch.backends.cudnn.benchmark=True                 
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# from model import net

# model_dir = './model_save/model_parameters.pth.tar'
# img_dir = './Fu.jpg'

# trans = transforms.Compose([ transforms.ToTensor(),
#                             transforms.Normalize((0.1307,), (0.3081,))
#                                       ])

# my_model = net()

# my_models = nn.DataParallel(net)
# my_model.load_state_dict(torch.load(model_dir)) 
# my_model = my_model.cuda()   #use gpu to run the model
# my_model.eval()

# img = Image.open(img_dir).convert('L')
# img = trans(img)  
# img = img.unsqueeze(0)

# output = my_model(img.cuda())
# _, preds = torch.max(outputs.data,1)
# #I use the gpu0 to run the model, so I should use '.cpu()' before using '.numpy()'
# a = preds.data.cpu().numpy() 
# print('preds:',img_name,' ',a)