import os
import tensorflow as tf
import utility.feature_fusion as ff
import numpy as np
import operator

#关于path 使用相对路径较为方便
AUDIO_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\mfcc"  #实验室路径
IMG_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\bottleneck"
TFRecord_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\TFRecord\\train.tfrecords"  # 实验室TFRecord path
CLASSMAP_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\TFRecord\\classmap.txt"

#存放数据个数
example_num = 30
sample_num = 3

#类别总数
class_num = 10

#第几个图片
num = 0

#第几个TFRecord文件
recordfilenum = 0


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#加载数据
def load_data(audio_path,img_path,onehot=None):
    features,labels=[],[]
    class_map = {} #标签字符与index对应关系字典
    audio_feature = ff.readText(ff.getDataInfo(audio_path), type='audio')
    img_feature = ff.readText(ff.getDataInfo(img_path), type='image')
    data = ff.direct_fusion(audio_feature, img_feature)
    for fusion in data:
        features.append(fusion['fusion_feature'])
        labels.append(fusion['classes'])
    if onehot is not None:
        for index,name in enumerate(labels):
            class_map[name] = int((index+1)/sample_num)  #3为类别中样本个数
        labels = []
        for fusion in data:
            onehot_init = np.zeros(class_num,dtype=np.int32)
            classes = fusion['classes']
            for key in class_map:
                if key == classes:
                    label_index = class_map[key]-1
                    onehot_init[label_index] = 1
            labels.append(onehot_init)
    return features,labels,class_map

def createTFRecord(filename,mapfile):
    cross_features, labels,class_map = load_data(AUDIO_PATH, IMG_PATH, onehot='true')
    # 输出TFRecord文件地址

    # 创建writer
    writer = tf.python_io.TFRecordWriter(filename)
    for data, label in zip(cross_features, labels):
        # 将样例转化为Example Protocol Buffer,并写入信息. 若不采用onehot编码，label的feature类型为int64
        print(data,label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'data': _bytes_feature(data.tobytes()),  # np.tobytes()
            'label': _bytes_feature(label.tobytes()),
        }))
        writer.write(example.SerializeToString())
    writer.close()

    #可选择是否保存class_map
    txtfile = open(mapfile, 'w+')
    for i in range(10):
        for key in class_map:
            if(class_map[key] == i):
                txtfile.writelines( str(key) + ":" + str(class_map[key]) + "\n")
    txtfile.close()

def read_and_decode(filename):
    #创建reader读取TFRecord()
    reader = tf.TFRecordReader()
    #创建队列维护输入文件列表
    filename_queue = tf.train.string_input_producer([filename],shuffle=False)
    #从文件中读出一个样例，也可使用read_up_to一次读取多个样例
    _,serialized_example = reader.read(filename_queue)

    #解析读入的一个样例，若解析多个，可用parse_example
    features = tf.parse_single_example(
        serialized_example,
        features={'data':tf.FixedLenFeature([],tf.string),
                  'label':tf.FixedLenFeature([],tf.string)}
    )

    #字符串解析
    data = tf.decode_raw(features['data'],tf.float32)
    data = tf.reshape(data,[7268,1]) #reshap为7268*1特征 in_width:7268 ,in_channel:1
    labels = tf.decode_raw(features['label'],tf.int32)
    labels = tf.reshape(labels,[10])
    return data,labels

def createBatch(filename,batchsize):
    data, labels = read_and_decode(filename)
    min_after_dequeue = 10 #?
    capacity = min_after_dequeue+3*batchsize

    #样本随机打乱
    data_batch,label_batch = tf.train.shuffle_batch([data,labels],
                                                    batch_size=batchsize,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
    return data_batch,label_batch

# if __name__ == "__main__":
#     createTFRecord(TFRecord_PATH,CLASSMAP_PATH)
#     image_batch, label_batch = createBatch(filename=TFRecord_PATH, batchsize=3)
#     with tf.Session() as sess:
#         initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#         sess.run(initop)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         try:
#             step = 0
#             while 1:
#                 _feature_batch, _label_batch = sess.run([image_batch, label_batch])
#                 step += 1
#                 print(step)
#                 print(_label_batch)
#                 print(_feature_batch)
#         except tf.errors.OutOfRangeError:
#             print(" trainData done!")
#
#         coord.request_stop()
#         coord.join(threads)

