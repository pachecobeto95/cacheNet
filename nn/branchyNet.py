"""
Author: Roberto Pacheco
Date: 06/08/2019
Last Modified: 06/08/2019
Objective: send a data to the edge node
"""

import torch, sys
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F


class AlexNet_Main_Branch(nn.Module):
	def __init__(self, n_classes=1000):
		super(AlexNet_Main_Branch, self).__init__()
		self.n_classes = n_classes
		self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
		self.lrn  = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.)

		self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)

		self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)

		self.conv4 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
		#Output fourth convolutional layer:13x13x384
		self.conv5 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
		#Output fifth convolutional layer:13x13x384
		#Input first FC layer:6x6x256
		self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
		self.fc2 = nn.Linear(in_features=4096, out_features=4096)
		self.fc3 = nn.Linear(in_features=4096, out_features=n_classes)

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
		x = F.dropout(x, p=0.5, training=True, inplace=False)

		x.view(-1, 6*6*256)

		#FC Layer
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=True, inplace=False)
		
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		if(self.n_classes == 2):
			return torch.sigmoid(x), [torch.sigmoid(x_exit1), torch.sigmoid(x_exit2)]

		else:

			return F.softmax(x, dim=1)
