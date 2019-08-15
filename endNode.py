"""
Author: Roberto Pacheco
Date: 06/08/2019
Last Modified: 06/08/2019
Objective: send a data to the edge node
"""

import os, config
from routines.sendData import sendImg


try:
	fileName = "1.jpg"
	imgFile = os.path.join(config.DIR_NAME, "datasets", fileName)
	url = config.URL_EDGE + "/api/edgearch/edge"
	sendImg(url, imgFile, fileName)
except Exception as e:
	print(e.args)