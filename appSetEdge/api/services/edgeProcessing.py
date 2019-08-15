from flask import jsonify, session, current_app as app
#from .featureExtractionCloud import FeatureExtractor
#from cache import LFUCache
import cv2, logging, os, pickle, h5py,requests, sys, config, time
import numpy as np, json
#import lfucache.lfu_cache as lfu_cache
from sklearn import linear_model
from scipy.misc import derivative






def checkCache(jsonData):
	try:
		with open(config.CACHE_FILE, "rb") as f:
			cache = pickle.load(f)
			print(jsonData['name'])
			print(cache.get(jsonData['name']))
		if(cache.get(jsonData['name']) == -1):
			print("n achou rs")
		else:
			print("achou")
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}





def setCache(fileImg):
	try:
		with open(config.CACHE_FILE, "r+b") as f:
			cache = pickle.load(f)
			cache.set(fileImg.filename, fileImg)
			pickle.dump(cache, f)


		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}


def uploadCloud(url, fileImg):
	try:
		imgPath = os.path.join(config.DIR_NAME, "appCloud", "api", "cloudDataset", fileImg.filename)
		files = {'file': ("uahahu.png", open(imgPath, 'rb'), 'image/x-png')}
		r = requests.post(url, files=files)

		if(r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s'%(r.status_code))

	except Exception as err:
		print(err.args)
		sys.exit()
	else:
		print("upload achieved")
