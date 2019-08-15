from flask import jsonify, session, current_app as app
import cv2, logging, os, pickle, h5py,requests, sys, config, time
import numpy as np, json
from sklearn import linear_model
from scipy.misc import derivative
from .cache import LFUCache
from ..classes import alexNet_conv
from torchvision import datasets, transforms
from PIL import Image
from torch.autograd import Variable


network = alexNet_conv.AlexNet_Conv()

def receiveData(fileImg):
	try:
		network.eval()
		imgPath = os.path.join(config.DIR_NAME, "appEdge", "api", "edgeDataset", fileImg.filename)
		__saveImage(fileImg, imgPath)
		url = config.URL_CLOUD + "/api/edgearch/cloud"
		
		#with open(config.CACHE_FILE, "rb") as f:
		#	cache = pickle.load(f)
		#dataRec = cache.get("a")
		img = __imageLoader(imgPath)
		features = network(img)
		uploadCloudFeatures(url, features.detach().numpy(), fileImg.filename)
		#if(dataRec == -1):
		#	uploadCloudFeatures(url, features.detach().numpy())
			#uploadCloud(url, fileImg)
		
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e.args)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}




def uploadCloud(url, fileImg):
	try:
		imgPath = os.path.join(config.DIR_NAME, "appEdge", "api", "edgeDataset", fileImg.filename)
		files = {'file': (fileImg.filename, open(imgPath, 'rb'), 'image/x-png')}
		
		r = requests.post(url, files=files)
		if(r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s'%(r.status_code))

	except Exception as err:
		print(err.args)
		sys.exit()
	else:
		print("upload achieved")

def uploadCloudFeatures(url, features, filename):
	try:
		data = {"features": features.tolist(), "filename":filename}
		r = requests.post(url, json=data)

		if(r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s'%(r.status_code))

	except Exception as err:
		print(err.args)
	else:
		print("upload achieved")

def setCache(data):
	try:
		with open(config.CACHE_FILE, "rb") as f:
			cache = pickle.load(f)
			print(cache.get(data["key"]))
			#cache = LFUCache(10)
			#cache.set(data["key"], data["value"])
			#pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}


def __imageLoader(imgPath):
	data_transforms = {'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])}
	img = Image.open(imgPath)
	img = data_transforms["test"](img).float()
	img = Variable(img, requires_grad=True)
	img = img.unsqueeze(0)
	return img

def __saveImage(imgFile, imgPath):
	if(imgFile):
		imgFile.save(imgPath)

