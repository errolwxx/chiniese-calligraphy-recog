import os, random, shutil

trainpath = 'dataset/train/'
words = os.listdir(trainpath)
def testGen(fileDir, labelDir):
    pathDir = os.listdir(fileDir)    #取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.025   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber*rate)#按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    # print(sample)
    for name in sample:
        if not os.path.exists(labelDir):
            os.mkdir(labelDir)
        shutil.move(fileDir + name, labelDir + name)
    return

if __name__ == '__main__':

    tarDir = "./dataset/test/"
    for word in words:
        fileDir = trainpath + word + "/"
        labelDir = tarDir + word + "/"
        testGen(fileDir, labelDir)
