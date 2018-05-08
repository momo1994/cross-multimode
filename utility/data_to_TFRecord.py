import os
import tensorflow as tf
import feature_fusion as ff

AUDIO_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\mfcc"  #实验室路径
IMG_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\bottleneck"

#存放数据个数
example_num = 30
sample_num = 3

#第几个图片
num = 0

#第几个TFRecord文件
recordfilenum = 0

#加载数据
def load_data(audio_path,img_path,onehot=None):
    features,labels=[],[]
    class_map = {}
    audio_feature = ff.readText(ff.getDataInfo(audio_path), type='audio')
    img_feature = ff.readText(ff.getDataInfo(img_path), type='image')
    data = ff.direct_fusion(audio_feature, img_feature)
    for fusion in data:
        features.append(fusion['fusion_feature'])
        labels.append(fusion['classes'])
    if onehot is not None:
        temp = 0
        for index,name in enumerate(labels):
            class_map[name] = int((index+1)/sample_num)  #3为类别中样本个数
    return features,labels

if __name__ == "__main__":
    cross_features, labels = load_data(AUDIO_PATH,IMG_PATH,onehot='true')
    #输出TFRecord文件地址
    filename = "/data/TFRecord/train.tfrecords"
    #创建writer
    writer = tf.python_io.TFRecordWriter(filename)
    # for index in range(example_num):
    #     cross_feature = features[index].tobytes()
    #     #将样例转化为Example Protocol Buffer,并写入信息
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'size':_int64_feature(7268),
    #         'label':_int64_feature()
    #     }))