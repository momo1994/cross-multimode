import os
import numpy
import mfcc_extract
import model.Multimode_Network
#根据需要修改路径
#AUDIO_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\mfcc"   #pc路径
#IMG_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\bottleneck"

AUDIO_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\mfcc"  #实验室路径
IMG_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\bottleneck"
#返回substr在str中第i次出现的位置
def findSubStr(substr, str, i):
    count = 0
    while i > 0:
        index = str.find(substr)
        if index == -1:
            return -1
        else:
            str = str[index+1:]
            i -= 1
            count = count + index + 1
    return count - 1

#返回dict{'path':**,'classes':**,'name':**}的集合
def getDataInfo(datapath):
    data_list=[]
    dataInfo_dict = {}
    for (root, dirs, files) in os.walk(datapath):
        for filename in files:
            path = os.path.join(root,filename)
            classes = path[findSubStr("\\", path, path.count("\\")-1)+1:findSubStr("\\", path, path.count("\\"))]
            name = path[findSubStr("\\", path, path.count("\\"))+1:findSubStr(".", path, 1)]
            dataInfo_dict['path'] = path
            dataInfo_dict['classes'] = classes
            dataInfo_dict['name'] = name
            dataInfo_dict_copy = dataInfo_dict.copy()
            data_list.append(dataInfo_dict_copy)
    return data_list

#读取特征数据
def readText(feature_dict_list,type):
    feature_list=[]
    feature_dict = {}
    for data_dict in feature_dict_list:
        feature = []
        feature_path = data_dict['path']
        if(type=='image'):
            feature = numpy.loadtxt(feature_path,delimiter=',').tolist()
        if(type == 'audio'):
            with open(feature_path) as f:
                read_string = f.read().replace("\n", "")
            for i in range(read_string.count("[")):
                dimension_x = read_string[findSubStr("[", read_string, i+1)+1:findSubStr("]", read_string, i+1)].split()
                feature.append(list(map(eval,dimension_x)))
        feature_dict['type'] = type
        feature_dict['feature'] = feature
        feature_dict['classes'] = data_dict['classes']
        feature_dict['name'] = data_dict['name']
        feature_dict_copy = feature_dict.copy()
        feature_list.append(feature_dict_copy)
    print("Read " +type +" data finish!")
    return feature_list

#直接拼接融合方法 -- 直接在构建的网络中调用
#融合特征size=(7268,)
def direct_fusion(feature_a,feature_b):
    fusion_feature_list = []
    fusion_feature_dict = {}

    for (audio,image) in zip(feature_a,feature_b):
        a = numpy.array(audio['feature']).reshape(-1) #audio的特征需要reshape成一维数组
        i = numpy.array(image['feature'])
        connection_feature = numpy.concatenate([a,i],axis=0)
        fusion_feature_dict['fusion_feature'] = connection_feature
        #print(connection_feature.shape)
        fusion_feature_dict['classes'] = audio['classes']
        fusion_feature_dict_copy = fusion_feature_dict.copy()
        fusion_feature_list.append(fusion_feature_dict_copy)
    print("Fusion feature finish!")
    return fusion_feature_list



if __name__ == "__main__":
    audio_data = getDataInfo(AUDIO_PATH)
    img_data = getDataInfo(IMG_PATH)
    audio_feature = readText(audio_data,type = 'audio')
    img_feature = readText(img_data, type='image')
    fusion = direct_fusion(audio_feature,img_feature)
    for i in fusion:
        result = i['fusion_feature'].get_shape()
        print("fusion size:" )


