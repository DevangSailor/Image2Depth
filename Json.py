import glob
import os
import csv
import cv2
"""
the objective of this mmethod is to 
convert all annotations insto their json format
"""
dicts={}
for file in glob.glob("./AirSim/*/*/"+"airsim_rec.txt"):
	in_txt = csv.reader(open(file, "rb"), delimiter = '\t')
	for row in in_txt:
		f=file.replace("airsim_rec.txt","images/")
		if ";" in row[-1]:
			image,pfm=row[-1].split(";")
			if os.path.isfile(f+image) and os.path.isfile(f+pfm):
				try:
					im=cv2..imread(f+image,0)
					if np.count_nonzero(im) < 0.15*1300*1680:
						dicts[f+image]=f+pfm
				except:
					print("entry removed")		
import json
with open('ImageAndDepthFile.json', 'w') as fp:
    json.dump(dicts, fp)	