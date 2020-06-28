from torchvision import datasets, transforms
import torch

batch_size = 64
LR = 0.01
EPOCH = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.hidden_1 = torch.nn.Linear(784, 128)
        self.hidden_2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden_1(x))
        x = torch.nn.functional.relu(self.hidden_2(x))
        out = self.output(x)
        return out


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])])

data_train = datasets.MNIST(root=r"./data/", transform=transform, train=True, download=True)
data_test = datasets.MNIST(root=r"./data/", transform=transform, train=False)
data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

classifier = Classifier().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

for i in range(EPOCH):
    for data in data_loader_train:
        x, y = data
        x, y = torch.autograd.Variable(x).to(device), torch.autograd.Variable(y).to(device)
        y_prediction = classifier.forward(x.view(x.shape[0], -1))
        loss = loss_function(y_prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

correct = 0
for data in data_loader_test:
    x, y = data
    x, y = torch.autograd.Variable(x).to(device), torch.autograd.Variable(y).to(device)
    y_prediction = classifier.forward(x.view(x.shape[0], -1))
    _, predicted = torch.max(y_prediction, 1)
    correct += torch.sum(predicted == y)

print("Correct: {}".format(correct))
print("Accuracy: {}".format(int(correct) / len(data_test)))

