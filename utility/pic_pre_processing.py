import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 

#问题：随机截取的区域图像有空白。
def distorted_bounding_box(image,height,width,bbox):
	if bbox is None:
		bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
	#转换图像张量的类型
	if image.dtype != tf.float32:
		image = tf.image.convert_image_dtype(image,dtype = tf.float32)
	#随机截取图像，减小需要关注的物体大小对图像识别算法的影响
	bbox_begin,bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes = bbox,min_object_covered=0.1)
	distorted_image = tf.slice(image,bbox_begin,bbox_size)
	#将随机截取的图像调整为神经网络输入层的大小
	distorted_image = tf.image.resize_images(distorted_image,size=[height,width],method=np.random.randint(4))

	return distorted_image


image_raw_data = tf.gfile.FastGFile('D:/workspace/python/cross-multimode/data/images/David/0001_1.png','rb').read()

with tf.Session() as sess:
	img_data = tf.image.decode_png(image_raw_data)
	print(img_data.eval())
	#bbox未指定
	bbox = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	for i in range(6):
	 	result = distorted_bounding_box(img_data,100,100,bbox = None)	
	 	#result = tf.image.convert_image_dtype(result,dtype = tf.uint16)
	 	#processed_data = tf.image.encode_png(result) 
	 	#with tf.gfile.GFile('D:/workspace/python/cross-multimode/data/images/David/00'+str(i)+'.png','wb') as f:
	 		#f.write(processed_data)	
	 	plt.imshow(result.eval())
	 	plt.show()