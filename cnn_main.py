#!/usr/bin/python
"""
"""
import os
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import cnn as myconv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#创建目录
def createdir(*args):
    for item in args:
        if not os.path.exists(item):#判断目录是否存在，不存在则创建目录
            os.makedirs(item)

#图像大小
IMGSIZE = 64
#一次训练的个数
batch_size = 10
savepath = './checkpoint/face.ckpt'
#获取大小以使图像为正方形
def getpaddingSize(shape):
    h, w = shape#获得高和宽
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

#图像处理
def dealwithimage(img, h=64, w=64):
    #img = cv2.imread(imgpath)
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img

#调整图片颜色
def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(float)#转float类型，Range in -1 to 1 or 0 to 1
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)#转回uint8，Range in 0 to 255
    return imgsrc

 #获取人脸,imgpath:图片路径  outdir：输出路径
def getface(imgpath, outdir):
    filename = os.path.splitext(os.path.basename(imgpath))[0]#返回路径的最终元素,将文件名和扩展名分开，得到文件名
    img = cv2.imread(imgpath)#读取图片
    classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')#OpenCV自带的分类器
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#RGB到gray,灰度图像加快detectMultiScale中的检测速度
    # 分类器对象调用,检测图片中的人脸部分，scaleFactor设置1.3每次搜索窗口依次扩大30%,minNeighbors构成检测目标的相邻矩形的最小个数设置为5
    # 以避免返回与人脸部分无关的矩形图片
    faces = classifier.detectMultiScale(gray_img, 1.3, 5)
    n = 0
    for f_x, f_y, f_w, f_h in faces:
        n += 1#第几张脸
        face = img[f_y:f_y+f_h, f_x:f_x+f_w]#获得在camera.read()读取的图片中的人脸部分, 即一开始得到的RGB图像中人脸部分
        # 可能现在不需要调整大小
        #face = cv2.resize(face, (64, 64))
        face = dealwithimage(face, IMGSIZE, IMGSIZE)#图像处理
        # for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
            # facetemp = relight(face, alpha, bias)
            # cv2.imwrite(os.path.join(outdir, '%s_%d_%d.jpg' % (filename, n, inx)), face)
        # 以filename_n命名图片，表示原来图片的名字filename加上第n张脸,将 face 存储到outdir路径的文件夹中
        cv2.imwrite(os.path.join(outdir, '%s_%d.jpg' % (filename, n)), face)

#从目录获取所有文件和文件夹,返回该目录中的所有.jpg文件
def getfilesinpath(filedir):
    for (path, dirnames, filenames) in os.walk(filedir):#遍历该目录，path：该目录路径  dirnames：文件夹集合 filenames：文件集合
        for filename in filenames:#遍历文件集合
            if filename.endswith('.jpg'):#如果是.jpg文件，则返回该文件路径
                yield os.path.join(path, filename)
        for diritem in dirnames:#遍历文件夹集合
            getfilesinpath(os.path.join(path, diritem))#从文件目录获取所有文件和文件夹

#生成人脸图片,将./image/trainfaces中的人脸图片添加到./image/trainimages
def generateface(pairdirs):
    #inputdir = ./image/trainimages     outputdir = ./image/trainfaces
    for inputdir, outputdir in pairdirs:
        for name in os.listdir(inputdir):#遍历包含目录中文件名称的列表,查询每个文件或者文件夹
            #得到文件或者文件夹的相对路径
            inputname, outputname = os.path.join(inputdir, name), os.path.join(outputdir, name)
            if os.path.isdir(inputname):#判断inputname是否为目录
                createdir(outputname)#创建目录
                for fileitem in getfilesinpath(inputname):#遍历所有返回的.jpg文件
                    getface(fileitem, outputname) #获取人脸

#读取图片
def readimage(pairpathlabel):
    imgs = []#图片列表
    labels = []#标签列表
    for filepath, label in pairpathlabel:#遍历文件路径-标签对
        for fileitem in getfilesinpath(filepath): #遍历filepath下的所有文件
            img = cv2.imread(fileitem)#从文件加载图像
            imgs.append(img)#添加到图片列表
            labels.append(label)#添加到标签列表
    return np.array(imgs), np.array(labels)#转换成ndarray类型返回

#使用one-hot编码,根据不同的人名，分配不同的onehot值。numlist : 数据列，one-hot编码之后非零的列
def onehot(numlist):
    b = np.zeros([len(numlist), len(numlist)])#获得全0矩阵
    b[numlist, numlist] = 1 #one-hot编码
    # b = np.zeros([len(numlist), max(numlist)+1])
    # b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()

#获取文件和标签
def getfileandlabel(filedir):
    # di = {}
    # for name in os.listdir(filedir):
    #     if os.path.isdir(os.path.join(filedir, name)):
    #         di[name] = os.path.join(filedir, name)
    #得到key为文件夹名字，value为文件夹路径的字典
    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, name))])
                    #for (path, dirnames, _) in os.walk(filedir) for dirname in dirnames])
    # dirnamelist, dirpathlist = di.keys(), di.values()
    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()#分别得到字典所有的键和字典所有的值
    indexlist = list(range(len(dirnamelist)))#获得索引
    return list(zip(dirpathlist, onehot(indexlist))), dict(zip(indexlist, dirnamelist))

def main(_):

    isneedtrain = False #判断是否需要训练
    if os.path.exists(savepath+'.meta') is False:#如果meta不存在，则表示需要进行训练
        isneedtrain = True
    if isneedtrain:
        generateface([['./image/trainimages', './image/trainfaces']])#生成人脸图片
        pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')#获取文件路径-标签对
        v_pathlabelpair,v_indextoname = getfileandlabel('./image/validate')#获取文件路径-标签对

        train_x, train_y = readimage(pathlabelpair)#读取图片,得到训练集，和答案
        v_train_x,v_train_y = readimage(v_pathlabelpair)#读取图片,得到验证集，和答案

        # train_x = train_x.astype(np.float32) / 255.0
        train_x = train_x.astype(np.float32)
        print('训练集大小 : %s', train_x.shape)
        myconv.train(train_x, train_y, v_train_x,v_train_y,savepath,batch_size)
    else:
        # testfromcamera(savepath)
        pass
        #print(np.column_stack((out, argmax)))

def testfromfile(chkpoint):
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # OpenCV自带的分类器
    pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')  # 获取文件路径-标签对
    output = myconv.cnnLayer(len(pathlabelpair))  # 得到神经网络的输出结果
    # predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output  # 预测值
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, chkpoint)  # 恢复模型

        # 读帧
        img = cv2.imread('./app/me.jpg')

        # 分类器对象调用,检测图片中的人脸部分，scaleFactor设置1.3每次搜索窗口依次扩大30%,minNeighbors构成检测目标的相邻矩形的最小个数设置为5
        # 以避免返回与人脸部分无关的矩形图片
        faces = haar.detectMultiScale(img, 1.3, 5)

        for f_x, f_y, f_w, f_h in faces:  # 遍历每个人脸部分的坐标和宽高
            face = img[f_y:f_y + f_h, f_x:f_x + f_w]  # 获得在camera.read()读取的图片中的人脸部分, 即一开始得到的RGB图像中人脸部分
            face = cv2.resize(face, (IMGSIZE, IMGSIZE))  # 调整图片的大小->64*64
            # could deal with face to train
            test_x = np.array([face])  # 图片->ndarray
            # test_x = test_x.astype(np.float32) / 255.0
            test_x = test_x.astype(np.float32)
            res = sess.run([predict, tf.argmax(output, 1)], \
                           feed_dict={myconv.x_data: test_x, \
                                      myconv.keep_prob_5: 1.0, myconv.keep_prob_75: 1.0})

            return str(indextoname[res[1][0]])

def testfromcamera(chkpoint):
    camera = cv2.VideoCapture(0)#打开摄像头
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#OpenCV自带的分类器
    pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')#获取文件路径-标签对
    output = myconv.cnnLayer(len(pathlabelpair))#得到神经网络的输出结果
    #predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output#预测值
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, chkpoint)#恢复模型
        n = 1
        while 1:
            if n <= 20000:
                print('处理第%s 图片.' % n)
                # 读帧
                success, img = camera.read()#解码并返回下一个视频帧

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#RGB到gray,灰度图像加快detectMultiScale中的检测速度
                # 分类器对象调用,检测图片中的人脸部分，scaleFactor设置1.3每次搜索窗口依次扩大30%,minNeighbors构成检测目标的相邻矩形的最小个数设置为5
                # 以避免返回与人脸部分无关的矩形图片
                faces = haar.detectMultiScale(gray_img, 1.3, 5)
                for f_x, f_y, f_w, f_h in faces:#遍历每个人脸部分的坐标和宽高
                    face = img[f_y:f_y+f_h, f_x:f_x+f_w]#获得在camera.read()读取的图片中的人脸部分, 即一开始得到的RGB图像中人脸部分
                    face = cv2.resize(face, (IMGSIZE, IMGSIZE))#调整图片的大小->64*64

                    test_x = np.array([face])#图片->ndarray
                    # test_x = test_x.astype(np.float32) / 255.0
                    test_x = test_x.astype(np.float32)
                    res = sess.run([predict, tf.argmax(output, 1)],\
                                   feed_dict={myconv.x_data: test_x,\
                                   myconv.keep_prob_5:1.0, myconv.keep_prob_75: 1.0})
                    print(res)

                    cv2.putText(img, indextoname[res[1][0]], (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                    img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                    n+=1
                cv2.imshow('img', img)
                key = cv2.waitKey(10) & 0xff
                if key == 27:#ESC退出
                    break
            else:
                break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(0)

