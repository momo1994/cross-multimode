import librosa
import librosa.display

AUDIO_PATH = ""
#获取声音地址
def get_path():


#提取mfcc特征
y,sr = librosa.load("D:/0001_1.wma",sr=None,duration=5.0)
mmfc_feature = librosa.feature.mfcc(y=y,sr=sr)
print(mmfc_feature.shape)
mmfc_string = ','.join(str(x) for x in mmfc_feature)
with open('D:/文档/跨模态检索实验/python/cross-multimode/mmfc/0001_1.txt','w') as mmfc_file:
    mmfc_file.write(mmfc_string)
