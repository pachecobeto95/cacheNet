import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

"""
YOLO paper: You Only Look Once: Unified, Real-Time Object Detection
Link: https://pjreddie.com/media/files/papers/yolo_1.pdf
"""
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

def create_modules(blockList):
	net_parameters = blockList[0]
	previous_layer = int(net_parameters["channels"])
	output_filters = []
	module_list = nn.ModuleList()
	for (idx, block) in enumerate(blockList[1:]):
		module = nn.Sequential()
		if(block["type"] == "convolutional"):
			activation_function = block['activation']
			n_filters = int(block['filters'])
			size = int(block["size"])
			stride = int(block["stride"])
			pad = int(block["pad"])
			try: 
				batch_normalize = block["batch_normalize"]
				bias = False
			except:
				batch_normalize = 0
				bias = True
			conv = nn.Conv2d(previous_layer, n_filters, kernel_size=size, 
				stride=stride, padding=pad, bias=bias)
			module.add_module("conv_{0}".format(idx), conv)
			if(batch_normalize):
				bn = nn.BatchNorm2d(n_filters)
				module.add_module("batch_norm_{0}".format(idx), bn)
			if(activation_function == "leaky"):
				leaky_function = nn.LeakyReLU(0.1, inplace = True)
				module.add_module("conv_{0}".format(idx), leaky_function)
		
		elif(block["type"] == "shortcut"):
			shortcut = EmptyLayer()
			module.add_module("conv_{0}".format(idx), leaky_function)		
		elif(block["type"] == "upsample"):
			stride = block["stride"]
			upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
			module.add_module("conv_{0}".format(idx), upsample)


		elif(block["type"] == "route"):
		
			block["layers"] = block["layers"].split(',')
			start = int(block["layers"][0])	
			try:
				end = int(block["layers"][1])
			except:
				end = 0

			if start > 0: 
				start = start - idx
			if end > 0:
				end = end - idx
			route = EmptyLayer()
			module.add_module("route_{0}".format(idx), route)
			if end < 0:
				n_filters = output_filters[idx + start] + output_filters[idx + end]
			else:
				n_filters= output_filters[idx + start]

		elif (block["type"] == "yolo"):
			mask = block["mask"].split(",")
			mask = [int(x) for x in mask]

			anchors = block["anchors"].split(",")
			anchors = [int(a) for a in anchors]
			anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
			anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors)
			module.add_module("Detection_{}".format(idx), detection)


		module_list.append(module)
		print(module_list)
		output_filters.append(n_filters)
		previous_layer = n_filters



def parse_cfg(cfg_file):
	file = open(cfg_file, 'r')
	lines = file.read().split('\n')
	lines = [x for x in lines if len(x) > 0]
	lines = [x for x in lines if x[0] != "#"]
	lines = [x.rstrip().lstrip() for x in lines] 
	
	blockDict = {}
	blockList = []

	for line in lines:
	    if line[0] == "[":               # This marks the start of a new blockDict
	        if len(blockDict) != 0:          # If blockDict is not empty, implies it is storing values of previous blockDict.
	            blockList.append(blockDict)     # add it the blockList list
	            blockDict = {}               # re-init the blockDict
	        blockDict["type"] = line[1:-1].rstrip()     
	    else:
	        key,value = line.split("=") 
	        blockDict[key.rstrip()] = value.lstrip()
	blockList.append(blockDict)
	return blockList


cfg_file = './yolov3.cfg'
blockList = parse_cfg(cfg_file)
create_modules(blockList)