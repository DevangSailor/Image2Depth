import tensorflow as tf
import numpy as np
#from models import 

def DCN3(inputs,KernelSize=7,FinalStride=2):
	TensorList=[]
	TensorList.append(inputs)

	for i in range(3):
		x = tf.layers.conv2d(inputs=tf.concat(TensorList,axis=-1),filters=8,padding='same',
			kernel_size=KernelSize,strides=1,dilation_rate=1,activation=tf.nn.relu,)
		TensorList.append(x)
	x =	tf.layers.conv2d(inputs=tf.concat(TensorList,axis=-1),filters=8,padding='same',
		kernel_size=KernelSize,strides=FinalStride,dilation_rate=1,activation=tf.nn.relu,)
	return x

def DCN5(inputs,KernelSize=5):
	TensorList=[]
	TensorList.append(inputs)

	for i in range(5):
		x = tf.layers.conv2d(inputs=tf.concat(TensorList,axis=-1),filters=16,padding='same',
			kernel_size=KernelSize,strides=1,dilation_rate=1,activation=tf.nn.relu,
			data_format='channels_last')
		TensorList.append(x)
	x =	tf.layers.conv2d(inputs=tf.concat(TensorList,axis=-1),filters=32,padding='same',
		kernel_size=KernelSize,strides=1,dilation_rate=1,activation=tf.nn.relu,
		data_format='channels_last')
	return tf.concat([x,inputs],axis=-1)	

def Upscaler(SkipInput,DirectInput,FineTuneKernerSize=5):
	x = tf.depth_to_space(input=DirectInput,block_size=2,data_format='NHWC')
	x = tf.layers.conv2d(inputs=x,filters=8,kernel_size=5,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)

	TensorList=[x,SkipInput]
	x = tf.concat(TensorList,axis=-1)

	x = DCN3(x,KernelSize=5,FinalStride=1)

	y = tf.layers.conv2d(inputs=x,filters=8,kernel_size=FineTuneKernerSize,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
	y = tf.layers.conv2d(inputs=y,filters=1,kernel_size=FineTuneKernerSize,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
	
	return y,x

def Build(inputs):
	x =  tf.layers.conv2d(inputs=inputs,filters=8,kernel_size=7,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
	x =  tf.layers.conv2d(inputs=x,filters=8,kernel_size=7,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
	
	X1 = DCN3(x)
	X2 = DCN3(X1)
	X3 = DCN3(X2)
	X4 = DCN3(X3)

	X5 = DCN5(X4)
	
	Y1,C1 = Upscaler(SkipInput=X3,DirectInput=X5)
	Y2,C2 = Upscaler(SkipInput=X2,DirectInput=C1)
	Y3,C3 = Upscaler(SkipInput=X1,DirectInput=C2)

	Y4 = tf.layers.conv2d(inputs=C3,filters=8,kernel_size=5,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
	Y4 = tf.depth_to_space(input=Y4,block_size=2,data_format='NHWC')
	Y4 = tf.layers.conv2d(inputs=Y4,filters=1,kernel_size=5,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
	return Y1,Y2,Y3,Y4

ph=tf.placeholder(tf.float32,shape=(1,512,512,3))
out=Build(ph)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	a=np.zeros((1,512,512,3))
	_,b,c,d=sess.run(out,feed_dict={ph:a})
	print(_.shape,b.shape,c.shape,d.shape)













