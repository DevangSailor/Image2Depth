import numpy as np
import cv2
import json
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
img = load_pfm(data[keys[0]])
print(img)         #pfm file name
