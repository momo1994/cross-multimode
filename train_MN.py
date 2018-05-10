import tensorflow as tf
import numpy as np
import sys
sys.path.append(r'D:\文档\跨模态检索实验\python\cross-multimode\utility')
sys.path.append(r'D:\文档\跨模态检索实验\python\cross-multimode\model')
import TFRecord
import feature_fusion
import Multimode_Network as MN


#Aim: 数据分出train vail test三部分
#     标签hot
#     loss构建
#     是否使用BP

#AUDIO_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\mfcc"   #pc路径
#IMG_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\bottleneck"

AUDIO_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\mfcc"  #实验室路径
IMG_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\bottleneck"

TFRecord_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\TFRecord\\train.tfrecords"  # 实验室TFRecord path

LABEL_NUM = 10
BATCH_SIZE = 3
LEARNING_RATE = 0.001
STEP = 3000

def train(logits,labels):
    #定义损失函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)
    correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)) #预测正确
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    return optimizer,cost,accuracy
    #show training process

if __name__ == "__main__":
    feature_batch,label_batch = TFRecord.createBatch(filename = TFRecord_PATH,batchsize=1)
    pred = MN.Multimode_Network(feature_batch)
    optimizer,cost,accuracy = train(logits=pred,labels=label_batch)
    #初始化tf参数
    initop = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(initop)
        coord = tf.train.Coordinator()#创建一个协调器，管理线程
        treads = tf.train.start_queue_runners(sess=sess,coord=coord)
        step = 0
        while step < STEP:
            step += 1
            #print(step)
            #_,loss,acc = sess.run([optimizer,loss,accuracy]) #命名冲突
            _, loss, acc = sess.run([optimizer,cost, accuracy])
            preds = sess.run(pred)  #softmax层输出
            if step % 100 == 0:
                print('STEP'+str(step) +":",acc)
                print('pred:',preds)
        print("Taining finish!")