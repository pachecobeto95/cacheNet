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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print('oi')
        sys.exit()
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
	test_loss /= len(testloader.dataset)
	test_losses.append(test_loss)
	print("Test set")
	print("Loss: %s, Accuracy: %s"%(test_loss, 100.0*float(correct)/len(test_data.dataset)))


def plot_result(train_losses, train_counter, test_losses, test_counter):
	figure = plt.figure()
	plt.plot(train_counter, train_losses, color='blue')
	plt.scatter(test_counter, test_losses, color='red')
	plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
	plt.xlabel('number of training examples seen')
	plt.ylabel('negative log likelihood loss')
	plt.show()


#transform = {x:transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) for x in ["training", "testing"]}
#datasetPath = os.path.join(config.DIR_NAME, "mnistasjpg")
#image_dataset = {x:datasets.ImageFolder(os.path.join(datasetPath, x), transform[x]) for x in ["training", "testing"]}
#data_loaders = {x:torch.utils.data.DataLoader(image_dataset[x], batch_size=4, shuffle=True, num_workers=4) for x in ["training", "testing"]}



#train_dataset = datasets.MNIST(DIR_NAME, download=True, train=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True)
#test_dataset = datasets.MNIST(DIR_NAME, download=True, train=False, transform=transform)
#testloader = torch.utils.data.DataLoader(test_dataset,batch_size=64, shuffle=True)



EPOCH = 5
learning_rate = 0.01
momentum = 0.5
log_interval = 10
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainloader.dataset) for i in range(EPOCH + 1)]

network = CNN()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

test(testloader, network, EPOCH)
for epoch in range(1, EPOCH + 1):
	train(trainloader, network, epoch)
	test(testloader, network, epoch)
