import tensorflow as tf
import numpy as np
import TFRecord
import model.Multimode_Network
import feature_fusion

#Aim: 数据分出train vail test三部分
#     标签hot
#     loss构建
#     是否使用BP

#AUDIO_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\mfcc"   #pc路径
#IMG_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\bottleneck"

AUDIO_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\mfcc"  #实验室路径
IMG_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\bottleneck"

LABEL_NUM = 10
BATCH_SIZE = 3
LEARNING_RATE = 0.001
STEP = 3000

def load_data(audio_path,img_path):
    audio_feature = readText(getDataInfo(audio_path), type='audio')
    img_feature = readText(getDataInfo(img_path), type='image')
    data = direct_fusion(audio_feature, img_feature)
    return data

def train(logits,labels):
    #定义输入输出
    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        7268,
        1
    ],name='x-input')

    label = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        LABEL_NUM
    ],name='label-input')

    #输出
    y = Multimode_Network.build(x)

    #定义损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=label))
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)
    correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1)) #预测正确
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    return optimizer,loss,accuracy
    #show training process

if __name__ == "__main__":
    train_filename = "/data/TFRecord/train.tfrecords" #TFRecord路径
    data_batch,label_batch = TFRecord.createBath(filename = train_filename,batchsize=3)
    pred = Multimode_Network.build(data_batch)
    optimizer,loss,accuracy = train(logits=pred,labels=label_batch)
    #初始化tf参数
    initop = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(initop)
        coord = tf.train.Coordinator()#创建一个协调器，管理线程
        treads = tf.train.start_queue_runners(sess=sess,coor=coord)
        step = 0
        while step < STEP:
            step += 1
            #print(step)
            _,loss,acc = sess.run([optimizer,cost,accuracy])
            if step % 100 == 0:
                print(loss,acc)
        print("Taining finish!")