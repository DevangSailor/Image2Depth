import tensorflow as tf
import numpy as np


def DCN3(inputs,KernelSize=7,FinalStride=2):
	"""
	this method is the immplemetation of the DCNN block
	args:
		inputs:			a tensor of format NHWC float tesor
		KernelSize:		size of side of the convolution kernel
		FinalStride:	stride of the last covolution process.
		  				FinalStride !=1 implies a downsample operation
	returns:
		a 4d NHWC flaot tensor
	"""
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
	"""
	this method is the immplemetation of the DCNN block
	args:
		inputs:			a tensor of format NHWC float tesor
		KernelSize:		size of side of the convolution kernel
		FinalStride:	stride of the last covolution process.
		  				FinalStride !=1 implies a downsample operation
	returns:
		a 4d NHWC flaot tensor
	"""

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
	"""
	this method declares the upsampler and also extract further feaures for higher scale
	output
	args:
		SkipInput:	sub-features from the encoder block,NHWC format float tensor
		DirectInput:NHWC format float tensor
		FineTuneKernerSize:kernel size of last covolution
	returns:
		y:	upsampled features NHWC twice in size of spatial resoluion of DirectInput
		x:	features NHWC float tensor 	

	"""
	x = tf.depth_to_space(input=DirectInput,block_size=2,data_format='NHWC')
	x = tf.layers.conv2d(inputs=x,filters=8,kernel_size=5,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)

	TensorList=[x,SkipInput]
	x = tf.concat(TensorList,axis=-1)

	x = DCN3(x,KernelSize=5,FinalStride=1)

	y = tf.layers.conv2d(inputs=x,filters=8,kernel_size=FineTuneKernerSize,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
	y = tf.layers.conv2d(inputs=y,filters=1,kernel_size=FineTuneKernerSize,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)

	return y,x

def Build(inputs):
	x =  tf.image.resize_images(images=inputs,size=(288,384))
	x =  tf.layers.conv2d(inputs=x,filters=8,kernel_size=7,strides=1,dilation_rate=1,activation=tf.nn.relu,padding='same',)
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
