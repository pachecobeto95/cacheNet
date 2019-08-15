import cv2, os, config, sys, glob
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt


#img = cv2.imread(os.path.join(config.DIR_NAME, "hymenoptera_data", "test", "ants", "10308379_1b6c72e180.jpg"))
#img = cv2.GaussianBlur(img, (95, 95), 0)
#cv2.imshow('a', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()






datasetPath = os.path.join(config.DIR_NAME, "mnistasjpg", "testing") 
resolutionList = range(5, 70, 10)

for resolution in resolutionList:
	for imgFile in glob.glob(os.path.join(datasetPath, "*", "*png")):
		try:
			os.mkdir(os.path.join(config.DIR_NAME, "mnistasjpg", "test_%s"%(resolution)))

		except:
			pass

		try: 
			os.mkdir(os.path.join(config.DIR_NAME, "mnistasjpg", "test_%s"%(resolution), imgFile.split("/")[-2]))
		except :
			pass

		img = cv2.imread(imgFile)
		img = cv2.GaussianBlur(img, (resolution, resolution), 0)
		#print(os.path.join(config.DIR_NAME, "mnistasjpg", "test_%s"%(resolution), imgFile.split("/")[-2], imgFile.split("/")[-2]))
		cv2.imwrite(os.path.join(config.DIR_NAME, "mnistasjpg", "test_%s"%(resolution), imgFile.split("/")[-2], imgFile.split("/")[-1]), img)