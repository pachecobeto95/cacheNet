import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

class AlexNet_Conv(nn.Module):
	def __init__(self):
		super(AlexNet_Conv, self).__init__()
		#Input Layer: 224x224x3
		self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
		#Output first convolutional layer: 55x55x96
		self.lrn  = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.)
		#Input second convolutional layer: 27x27x96
		self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
		#Output second convolutional layer: 27x27x256
		#Input third convolutional layer: 13x13x256		
		self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
		#Output third convolutional layer: 13x13x384
		#Input fourth convolutional layer: 13x13x384
		self.conv4 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
		#Output fourth convolutional layer:13x13x384
		self.conv5 = nn.Conv2d(192, 256, kernel_size=3, padding=1)

	def forward(self, x):
		#First Conv Layer
		x = F.relu(self.conv1(x))
		x = self.lrn(x)
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		#Second Conv Layer
		x = F.relu(self.conv2(x))
		x = self.lrn(x)
		x = F.max_pool2d(x, kernel_size=3, stride=2)

		#Third, fourth and fifth Conv Layers 
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		x.view(-1, 5*5*256)
		return x