import torch, os, sys, config
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(train_data, model, epoch):
	model.train()
	error = nn.CrossEntropyLoss()
	for batch_idx, (data, target) in enumerate(train_data):
		data = Variable(data).float()
		target = Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = error(output, target)
		loss.backward()
		optimizer.step()
		if((batch_idx % log_interval) == 0):
			print("Epoch: %s, Loss: %s"%(epoch, loss.item()))

def test(test_data, model, epoch):
	model.eval()
	test_loss = 0
	correct = 0
	error = nn.CrossEntropyLoss()
	with torch.no_grad():
		for (data, target) in test_data:
			data = Variable(data).float()
			target = Variable(target)
			output = model(data)
			test_loss += error(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	print("Test set")
	print("Loss: %s, Accuracy: %s"%(test_loss, 100.0*float(correct)/len(test_data.dataset)))
	return test_losses

def plot_result(train_losses, train_counter, test_losses, test_counter):
	figure = plt.figure()
	plt.plot(train_counter, train_losses, color='blue')
	plt.scatter(test_counter, test_losses, color='red')
	plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
	plt.xlabel('number of training examples seen')
	plt.ylabel('negative log likelihood loss')
	plt.show()

def plot_resolution_result(results_list, resolutionList, test_counter):
	plt.plot(test_counter, results_list[0])
	plt.plot(test_counter, results_list[1])
	plt.plot(test_counter, results_list[2])
	plt.plot(test_counter, results_list[3])
	plt.plot(test_counter, results_list[4])
	plt.plot(test_counter, results_list[5])
	plt.legend(['Blur = %s'%resolutionList[0], 
		'Blur = %s'%resolutionList[1], 
		'Blur = %s'%resolutionList[2], 
		'Blur = %s'%resolutionList[3],
		'Blur = %s'%resolutionList[4],
		'Blur = %s'%resolutionList[5]], loc='upper left')
	plt.savefig('./results/resolution_results')
	plt.show()





EPOCH = 5
learning_rate = 0.01
momentum = 0.5
log_interval = 10

network = CNN()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
resolutionList = [0, 5, 15, 25, 45, 65]

datasetPath = os.path.join(config.DIR_NAME, "mnistasjpg")
transform = {x:transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) for x in ["training", "testing"]}
train_dataset = datasets.ImageFolder(os.path.join(datasetPath, "training"), transform["training"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(EPOCH + 1)]
test_resolution_list = []


for resolution in resolutionList:
	test_dataset = datasets.ImageFolder(os.path.join(datasetPath, "test_%s"%(resolution)), transform["training"])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

	test(test_loader, network, EPOCH)
	
	for epoch in range(1, EPOCH + 1):
		train(train_loader, network, epoch)
		test_resolution = test(test_loader, network, epoch)
		print("ETA")
		print(test_resolution)
		test_resolution_list.append(test_resolution)

plot_resolution_result(test_resolution_list, resolutionList, test_counter)


