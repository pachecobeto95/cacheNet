import os, cv2, glob, sys, config
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import pandas as pd
#import seaborn as sbn
#import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch
from torch.autograd import Variable
from nn import exitLearnTest, alexnet

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
		self.fc1 = nn.Linear(28*28*64, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 2)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)

		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)

		x = F.relu(self.conv3(x))
		x = F.max_pool2d(x, 2)

		x = x.view(1, 28*28*64)

		x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		x = self.fc2(x)
		x = F.logsigmoid(x)
		return x



class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
 

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return probas

def statsDataset(train_dataset):
	df = pd.DataFrame(columns=['height', 'width', 'red', 'black', 'green'])
	transform = transforms.Compose([transforms.ToTensor()])
	train_set = datasets.ImageFolder(train_dataset, transform = transform)
	mean = 0.0
	heightList = []
	widthList = []
	for img, _ in train_set:
		mean += img.mean([1,2])
		heightList.append(img.shape[0])
		widthList.append(img.shape[1])

	df['height'] = heightList
	df['width'] = widthList
	df = pd.read_pickle("./dimensionsDataset")
	sbn.distplot(df["height"], hist=False, kde=True, color = 'darkblue', kde_kws={'linewidth': 4})
	sbn.distplot(df["width"], hist=False, kde=True, color = 'red', kde_kws={'linewidth': 4})
	print(mean)


def loadDatasets(datasetPath):
	transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean=[0.4883, 0.4551, 0.4174], std=[0.229, 0.224, 0.225])])
	image_datasets = {x: datasets.ImageFolder(os.path.join(datasetPath, x), transform = transform) for x in ['train', 'test']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True) for x in ['train', 'test']}
	return dataloaders

def exit_loss(weightList, output_side_branch, target, error):

	loss = np.array([[error(output, target)] for output in output_side_branch])
	loss_exit_branch = np.dot(weightList, loss)
	return loss_exit_branch.item(0)


def train(model, train_loader, epoch):
	model.train()
	error = nn.CrossEntropyLoss()
	for batch_idx, (data, target) in enumerate(train_loader):
		data = Variable(data).float()
		target = Variable(target)
		output_principal_branch, output_side_branch = model(data)
		loss_side_branch = exit_loss(config.weight_branchNet, output_side_branch, target, error)
		loss_principal = error(output_principal_branch, target)
		optimizer.zero_grad()
		loss_side_branch.backward(retain_graph=True)
		loss_principal.backward()
		optimizer.step()
		if(batch_idx%log_interval == 0):
			print("epoch: %s, Loss_Main: %s, Loss_Side: %s"%(epoch, loss_principal.item(), 
				loss_side_branch.item()))
		#if(batch_idx == 10):
		#	sys.exit()



def test(model, test_loader, epoch):
	model.eval()
	test_loss = 0
	correct = 0
	error = nn.CrossEntropyLoss()
	with torch.no_grad():
		for (data, target) in test_loader:
			data = Variable(data).float()
			target = Variable(target)
			output = model(data)
			test_loss += error(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	print("Loss: %s, Accuracy: %s"%(test_loss, 100.0*(float(correct)/len(test_loader.dataset))))

n_classes = 2
log_interval = 100
NUM_EPOCHS = 20
datasetPath = os.path.join(config.DIR_NAME, 'datasets', 'cat-and-dog')
dataloaders = loadDatasets(datasetPath)

net = exitLearnTest.AlexNet_BranchyNet(n_classes) 

optimizer = optim.SGD(net.parameters(), lr=alexnet.LR_INIT, momentum=alexnet.MOMENTUM, weight_decay=alexnet.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=alexnet.LR_DECAY)
for epoch in range(1, NUM_EPOCHS + 1):
	train(net, dataloaders['train'], epoch)
	test(net, dataloaders['test'], epoch)
	scheduler.step()
