import librosa
import librosa.display

import os

AUDIO_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\data\\audio"

SAVE_PATH = "D:\\文档\\跨模态检索实验\\python\\cross-multimode\\mmfc"


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
    dataInfo_dict={}
    for (root, dirs, files) in os.walk(AUDIO_PATH):
        for filename in files:
            path = os.path.join(root,filename)
            classes = path[findSubStr("\\", path, 7)+1:findSubStr("\\", path, 8)]
            name = path[findSubStr("\\", path, 8)+1:findSubStr(".", path, 1)]
            dataInfo_dict['path'] = path
            dataInfo_dict['classes'] = classes
            dataInfo_dict['name'] = name
            data_list.append(dataInfo_dict)
    return data_list


print(getDataInfo(AUDIO_PATH))

# for (root, dirs, files) in os.walk(AUDIO_PATH):
#      for filename in files:
#          # 提取mfcc特征
#          read_path = os.path.join(root,filename)
#          y, sr = librosa.load(read_path, sr=None, duration=5.0)
#          mmfc_feature = librosa.feature.mfcc(y=y, sr=sr)
#          start_index = findSubStr("\\", read_path, 7)
#          end_index = findSubStr(".", read_path,1)
#          save_path = os.path.join(SAVE_PATH,read_path[start_index+1:end_index])+'.txt'
#          print(save_path)
#          mmfc_string = ','.join(str(x) for x in mmfc_feature)
#          with open(save_path, 'w') as mmfc_file:
#              mmfc_file.write(mmfc_string)



