import inspect
import os

import tensorflow as tf
import numpy as np
import time

#Q1:input,output参数值如何定义/计算
#Q2:最后一层最为logist，如何与label(label为string类型)做差得到loss

#Multimode_Network网络结构
def Multimode_Network(input):
    #input: [3,7268,1] ,batch=3
    conv1_1 = conv1d_layer(input,1,12,"conv1_1")#(3,3634,12)
    conv1_2 = conv1d_layer(conv1_1,12,12,"conv1_2") #(3,1817,12)
    pool1 = avg_pool(conv1_2,"pool1") #(3,1817,12)

    conv2_1 = conv1d_layer(pool1, 12, 24, "conv2_1") #(3,909,24)
    conv2_2 = conv1d_layer(conv2_1, 24, 24, "conv2_2")#(3,455,24)
    pool2 = avg_pool(conv2_2, 'pool2') #(3,455,24)

    # flatten
    shp = pool2.get_shape()
    flattened_shape = shp[1].value * shp[2].value
    resh1 = tf.reshape(pool2, [-1, flattened_shape], name="resh1") #(3,10920)

    #fully connected
    fc3 = fc_layer(resh1,500,"fc3") #(3,500)
    fc3_drop = tf.nn.dropout(fc3,0.5,name="fc_drop4") #(3,500)

    fc4 = fc_layer(fc3_drop,10,"fc5") #(3,10)
    fc4_drop = tf.nn.dropout(fc4,0.5,name="fc_drop6") #(3,10)

    prob = tf.nn.softmax(fc4_drop, name="prob") #softmax输出size与fc_drop6一样 #(3,10)
    return prob




#定义卷积操作 filter通道数与输入的in_channels相同，out_channels为filter个数
def conv1d_layer(bottom,in_channels,out_channels, name):
    with tf.variable_scope(name):
        #filter [filter_width, in_channels, out_channels]
        filter = tf.get_variable('filter',[3,in_channels,out_channels],initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv = tf.nn.conv1d(bottom, filter,stride=2, padding='SAME')

        conv_biases = tf.get_variable('biases_conv1d', [out_channels] ,initializer=tf.zeros_initializer())
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

#定义池化层
def avg_pool(bottom,name):
    return tf.nn.pool(bottom,window_shape=[2],pooling_type='AVG',padding='SAME',name=name)

#定义全连接层
def fc_layer(bottom, out_size, name):
    with tf.variable_scope(name):
        in_size = bottom.get_shape()[-1].value #扁平化后的值
        x = bottom

        weights = tf.get_variable('weight',[in_size,out_size],initializer=tf.truncated_normal_initializer(stddev=0.1)) #是否要加batch? 不加
        biases = tf.get_variable('biases_fc', [out_size] ,initializer=tf.zeros_initializer())

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc