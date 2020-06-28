import numpy as np
import torch

LR = 0.01
EPOCH = 1000


class MLP:
    def __init__(self, input_size, output_size, hidden_size):
        self.w1 = np.random.normal(size=(hidden_size, input_size))
        self.w2 = np.random.normal(size=(output_size, hidden_size))
        self.b1 = np.random.normal(size=(hidden_size, 1))
        self.b2 = np.random.normal(size=(output_size, 1))
        self.input = np.zeros(1)
        self.h1 = np.zeros(1)
        self.out = np.zeros(1)

    def forward(self, x):
        self.input = np.atleast_2d(x).T
        self.h1 = MLP.sigmoid(np.dot(self.w1, self.input) + self.b1)
        self.out = MLP.sigmoid(np.dot(self.w2, self.h1) + self.b2)
        return self.out

    def backward(self, y, lr=LR):
        d_l2 = self.out - y
        d_w2 = np.dot(d_l2, self.h1.T)
        d_b2 = d_l2

        d_l1 = np.dot(self.w2.T, d_l2) * MLP.d_sigmoid(np.dot(self.w1, self.input) + self.b1)
        d_w1 = np.dot(d_l1, self.input.T)
        d_b1 = d_l1

        self.w1 -= lr * d_w1
        self.w2 -= lr * d_w2
        self.b1 -= lr * d_b1
        self.b2 -= lr * d_b2

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return MLP.sigmoid(x) * (1 - MLP.sigmoid(x))


class MLPTorch(torch.nn.Module):
    def __init__(self):
        super(MLPTorch, self).__init__()
        self.fc1 = torch.nn.Linear(3, 3)
        self.fc2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


x = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
y_choice = [-1, 1]
y = [1, 0, 0, 1, 0, 1, 1, 0]
x = torch.from_numpy(np.asarray(x)).float()
y = torch.from_numpy(np.asarray(y).reshape((8, 1))).float()

mlp = MLPTorch()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
for i in range(EPOCH):
    y_prediction = mlp.forward(x)
    loss = loss_function(y_prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for i in x:
    print(1 if mlp.forward(i).detach().numpy() > 0.5 else -1)

for i in mlp.state_dict().items():
    print(i)
