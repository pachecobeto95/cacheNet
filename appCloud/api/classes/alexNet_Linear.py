import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F


class AlexNet_Linear(nn.Module):
	def __init__(self, n_classes=1000):
		super(AlexNet_Linear, self).__init__()

		self.fc1 = nn.Linear(in_features=5*5*256, out_features=4096)
		self.fc2 = nn.Linear(in_features=4096, out_features=4096)
		self.fc3 = nn.Linear(in_features=4096, out_features=n_classes)

	def forward(self, x):
		x = F.dropout(x, p=0.5, training=True, inplace=False)
		x = x.view(-1, 5*5*256)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=True, inplace=False)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)

