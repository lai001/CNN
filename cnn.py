#!/usr/bin/python
#coding=utf-8
''' 卷积神经网络'''

import os
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
#通道
CHANNELS = 3
#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 3
#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3
#第三层卷积层的尺寸和深度
CONV3_DEEP = 64
CONV3_SIZE = 3
#全连接层的节点个数
FC_SIZE=512
#图像尺寸大小
SIZE = 64
#训练次数
STEPS=10
#一次训练的个数
batch_size = 10
x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, CHANNELS])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    #过滤器权重变量
    init = tf.random_normal(shape, stddev=0.01)
    #init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    #建立偏置项变量
    init = tf.random_normal(shape)
    #init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def conv2d(x, W):
    #使用边长为3，深度为32的过滤器，过滤器移动的步长为1.且使用全0填充
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    #选用最大池化层，池化层过滤器边长为2，使用全0填充且移动的步长为2
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    ''' dropout避免过拟合，使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep大小'''
    return tf.nn.dropout(x, keep)

#定义卷积神经网络的向前传播过程
def cnnLayer(classnum):
    # 第一层卷积层的尺寸为3*3 ，输入的颜色通道为3 ，深度为32
    W1 = weightVariable([CONV1_SIZE, CONV1_SIZE, CHANNELS, CONV1_DEEP])#过滤器权重变量
    b1 = biasVariable([CONV1_DEEP])#偏置项
    #卷积层的向前传播过程,先使用conv2d卷积函数，再使用RELU激活函数线性变换->非线性变换,使用边长为3，深度为32的过滤器
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)

    pool1 = maxPool(conv1)#池化层的向前传播过程,输出32*32*32的矩阵


    # 第二层
    W2 = weightVariable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])#过滤器
    b2 = biasVariable([CONV2_DEEP])#偏置项
    # 卷积层的向前传播过程,先使用conv2d卷积函数，再使用RELU激活函数线性变换->非线性变换,使用边长为3，深度为64的过滤器
    conv2 = tf.nn.relu(conv2d(pool1, W2) + b2)
    pool2 = maxPool(conv2)#池化层的向前传播过程，输出16*16*64的矩阵


    # 第三层
    W3 = weightVariable([CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP])#过滤器
    b3 = biasVariable([CONV3_DEEP])#偏置项
    # 卷积层的向前传播过程,先使用conv2d卷积函数，再使用RELU激活函数线性变换->非线性变换,使用边长为3，深度为64的过滤器
    conv3 = tf.nn.relu(conv2d(pool2, W3) + b3)
    pool3 = maxPool(conv3)#池化层的向前传播过程，输出8*8*64的矩阵

    pool3_shape = pool3.get_shape().as_list()  # 得到矩阵的维度
    nodes = pool3_shape[1] * pool3_shape[2] * pool3_shape[3]  # 计算将矩阵拉直成向量之后的长度，长度为矩阵的长宽和深度的乘积

    # 全连接层
    Wfc = weightVariable([nodes, FC_SIZE])
    bfc = biasVariable([FC_SIZE])
    pool3_reshape = tf.reshape(pool3, [-1, nodes])
    # Wfc = weightVariable([nodes, FC_SIZE])#全过滤器
    # bfc = biasVariable([FC_SIZE])#偏置项
    # pool3_reshape = tf.reshape(pool3, [pool3_shape[0], nodes])#将池化层的输出变成一个batch的向量
    fc = tf.nn.relu(tf.matmul(pool3_reshape, Wfc) + bfc)#全连接层的向前传播过程
    dropfc = dropout(fc, keep_prob_75)# 避免过拟合，随机让某些权重不更新

    # 输出层
    Wout = weightVariable([FC_SIZE, classnum])#这一层的输入为一组长度为512的向量，输出一组长度为classnum的向量
    bout = weightVariable([classnum])#偏置项
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropfc, Wout), bout)#输出层的向前传播过程

    return out
#训练
def train(train_x, train_y, v_train_x, v_train_y, tfsavepath, batch_size=None):

    out = cnnLayer(train_y.shape[1])
    validate_out = cnnLayer(v_train_y.shape[1])

    #使用softmax的多分类,并求交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)#使用优化器,使cross_entropy最小
    #判断每行中的最大值是否相等，并将结果转换成float类型,使用tf.argmax（，axis=1）时，两个比较时，行数必须相等
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(validate_out, 1), tf.argmax(v_train_y, 1)), tf.float32))

    saver = tf.train.Saver()#持久化
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#初始化全局变量

        # num_batch = len(train_x) // batch_size#以batch_size为单位，计算数量
        while batch_size<len(train_x):#以初始batch_size的大小为基准，每轮加一直到len(train_x)能整除batch_size
            if len(train_x)%batch_size==0:
                break
            else:batch_size+=1
        num_batch = len(train_x) // batch_size#以batch_size为单位，计算数量
        for n in range(STEPS):#训练10轮
            # r = np.random.permutation(len(train_x))
            # train_x = train_x[r, :]
            # train_y = train_y[r, :]
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                _, loss = sess.run([train_step, cross_entropy],\
                                   feed_dict={x_data:batch_x, y_data:batch_y,
                                              keep_prob_5:0.75, keep_prob_75:0.75})
                # print(n,num_batch,i)
                # print(n * num_batch + i)
                # print(i, loss)#训练次数和损失值

            # 获取测试数据的准确率
            acc = accuracy.eval({x_data:v_train_x,y_data:v_train_y, keep_prob_5:1.0, keep_prob_75:1.0})
            # print('accuracy is %s' % (n,acc))
        saver.save(sess, tfsavepath)

#验证
def validate(test_x, tfsavepath):
    output = cnnLayer(2)
    #predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, tfsavepath)
        res = sess.run([predict, tf.argmax(output, 1)],
                       feed_dict={x_data: test_x,
                                  keep_prob_5:1.0, keep_prob_75: 1.0})
        return res

if __name__ == '__main__':
    pass
