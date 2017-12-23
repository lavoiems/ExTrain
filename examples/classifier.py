from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# data loader
def data_loader():
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                              shuffle=True, num_workers=2)


    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


class Classifier(object):
    def __init__(self, net):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.loss = []

    def train(self, data):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = self.criterion(outputs, labels)
        self.loss.append(loss)
        loss.backward()
        self.optimizer.step()

    def evaluate(self, data):
        inputs, labels = data
        inputs = Variable(inputs)

        outputs = self.net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum()
        return correct / total


# Experiment
if __name__ == '__main__':
    # Data
    train_loader, test_loader = data_loader()

    # Model
    net = Net()
    net = net.cuda()
    classifier = Classifier(net)
    for i in range(10):
        classifier.evaluate(test_loader.dataset)
        for data in train_loader:
            classifier.train(data)
