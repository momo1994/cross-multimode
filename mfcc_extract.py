import librosa
import librosa.display

import os

#批量提取音频数据mfcc特征
#2018-03-28
#shao

#根据需要修改路径
AUDIO_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\data\\audio"

SAVE_PATH = "D:\\Workspaces\\GitHub\\cross-multimode\\mfcc"


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


def getDataInfo(datapath):
    data_list=[]
    dataInfo_dict = {}
    for (root, dirs, files) in os.walk(AUDIO_PATH):
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



if __name__=="__main__":
    audio_data = getDataInfo(AUDIO_PATH)
    for data_dict in audio_data:
        read_path = data_dict['path']
        y, sr = librosa.load(read_path, sr=None, duration=5.0)
        mmfc_feature = librosa.feature.mfcc(y=y, sr=sr)
        print(mmfc_feature)
        break
        # 保存路径
        save_dir = os.path.join(SAVE_PATH,data_dict['classes'])
        save_path = os.path.join(SAVE_PATH,data_dict['classes'],data_dict['name'])+'.txt'
        #判断路径是否存在，不存在则创建该路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        mmfc_string = ','.join(str(x) for x in mmfc_feature)
        with open(save_path, 'w') as mmfc_file:
            mmfc_file.write(mmfc_string)





