""" face detect
使用摄像头获得人脸图片
"""

import os
import random
import numpy as np
import cv2
#创建文件夹，如果该文件夹不存在
def createdir(*args):

    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMGSIZE = 64



#调整图片颜色，提高模型的泛化能力,imgsrc:图片  alpha:参数1  bias:偏置项
def Color_adjustment(imgsrc, alpha=1, bias=0):

    imgsrc = imgsrc.astype(float)#转float类型，Range in -1 to 1 or 0 to 1
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)#转回uint8，Range in 0 to 255
    return imgsrc
#outdir:存储人物图像的路径---相对路径
def getfacefromcamera(outdir):
    createdir(outdir)#创建文件夹
    camera = cv2.VideoCapture(0)#打开摄像头
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#OpenCV自带的分类器
    n = 1
    while 1:
        if n <= 200:#循环200次，拍下200张图片
            print('处理第 %s 张图片.' % n)
            # 读帧
            success, img = camera.read()#解码并返回下一个视频帧

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#RGB到gray,灰度图像加快detectMultiScale中的检测速度
            # 分类器对象调用,检测图片中的人脸部分，scaleFactor设置1.3每次搜索窗口依次扩大30%,minNeighbors构成检测目标的相邻矩形的最小个数设置为5
            #以避免返回与人脸部分无关的矩形图片
            faces = classifier.detectMultiScale(image=gray_img, scaleFactor=1.3, minNeighbors=5)
            for f_x, f_y, f_w, f_h in faces: #遍历每个人脸部分的坐标和宽高
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]#获得在camera.read()读取的图片中的人脸部分, 即一开始得到的RGB图像中人脸部分
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))#调整图片的大小->64*64

                # face = Color_adjustment(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)#以数字命名图片，将 face 存储到outdir路径的文件夹中
                cv2.putText(img, 'figure', (f_x, f_y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示文字
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n+=1
            cv2.imshow('img', img)
            key = cv2.waitKey(10) & 0xff#在某些系统中，返回的键盘值可能不是ASCII编码的，所以通过与运算只取字符最后一个字节
            if key == 27:#按下键盘的ESC退出
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # name = input('输入你的名字: ')
    name='lmc'
    getfacefromcamera(os.path.join('./image/trainfaces', name))

