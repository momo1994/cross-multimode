import inspect
import os

import tensorflow as tf
import numpy as np
import time

#Q1:input,output参数值如何定义/计算
#Q2:最后一层最为logist，如何与label(label为string类型)做差得到loss

class Multimode_Network:
    #Multimode_Network网络结构
    def build(self,input):
        #input: [3,7268,1] ,batch=3
        self.conv1_1 = self.conv1d_layer(input,1,12,"conv1_1")
        self.conv1_2 = self.conv1d_layer(self.conv1_1,12,"conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2,"pool1") #3643

        self.conv2_1 = self.conv_layer(self.pool1, 12, 24, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 12, 24, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')#1821.5

        self.fc3 = self.fc_layer(self.pool2,1822,500,"fc3")
        self.relu4 = tf.nn.relu(self.fc4,"relu4")
        self.prob = tf.nn.softmax(self.relu4, name="prob")




    #定义卷积操作 filter通道数与输入的in_channels相同，out_channels为filter个数
    def conv1d_layer(self, bottom,in_channels,out_channels, name):
        with tf.variable_scope(name):
            #filter [filter_width, in_channels, out_channels]
            filter = self.get_variable('filter',[3,in_channels,out_channels],initializer=tf.truncated_normal_initializer(stddev=0.1))

            conv = tf.nn.conv1d(bottom, filter,stride=2, padding='SAME')

            conv_biases = self.get_variable('biases_conv1d', [out_channels] ,initializer=tf.zeros_initializer())
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    #定义池化层
    def avg_pool(self,bottom,name):
        return tf.nn.pool(bottom,window_shape=[2],pooling_type='AVG',padding='SAME',name=name)

    #定义全连接层
    def fc_layer(self, bottom,in_size, out_size, name):
        with tf.variable_scope(name):
            #flatten
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_variable('weight',[in_size,out_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = self.get_variable('biases_fc', [out_size] ,initializer=tf.zeros_initializer())

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc