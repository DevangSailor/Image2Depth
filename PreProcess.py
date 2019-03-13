import numpy as np
import cv2
import json
from Prototype import *
import cv2
def load_pfm(fn):
    
    if fn.endswith(".pfm"):
        fid = open(fn, "rb")
    else:
        print("No pfm file! \n")
        return
    
    raw_data = fid.readlines()
    fid.close()

    cols = int(raw_data[1].strip().split(" ")[0])
    rows = int(raw_data[1].strip().split(" ")[1])
    
    del raw_data[2]     #data like size and type are removed before constructing the image
    del raw_data[1]
    del raw_data[0]
    
    image = np.fromstring("".join(raw_data), dtype=np.float32)      #Image will be a 1D long array
    del raw_data
    image = image.reshape(rows, cols)
    return image

with open('ImageAndDepthFile.json') as f:
    data = json.load(f)
keys=data.keys()
InputPlaceholder=tf.placeholder(tf.float32,shape=(None,None,None,3))

DP4=tf.placeholder(tf.float32,shape=(None,None,None,1))
DP3=tf.placeholder(tf.float32,shape=(None,None,None,1))
DP2=tf.placeholder(tf.float32,shape=(None,None,None,1))
DP1=tf.placeholder(tf.float32,shape=(None,None,None,1))



O4,O3,O2,O1=Build(ph)

LossTensor=tf.losses.mean_squared_error(O4,DP4)+
           tf.losses.mean_squared_error(O3,DP3)+
           tf.losses.mean_squared_error(O2,DP2)+
           tf.losses.mean_squared_error(O1,DP1)
optimizer=tf.train.AdamOptimizer().minimize(LossTensor)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for key in keys:
        image=cv2.imread(key)
        GT4=load_pfm(data[key])
        GT3=GT4[::2,::2]
        GT2=GT2[::4,::2]
        GT1=GT1[::8,::8]
        Loss,_=sess.run([LossTensor,optimizer],feed_dict={DP1:GT1,DP2:GT2,DP3:GT3,DP4:GT4,InputPlaceholder:image})
        print(Loss)





