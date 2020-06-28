import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# Using standard normal
DELTA = 1
EPOCH = 1000
LR = 0.01


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_prediction = self.linear(x)
        return y_prediction


# y = x + n
n = np.random.normal(loc=0, scale=1, size=500)
x = torch.from_numpy(np.asarray([[random.uniform(-10, 10)] for i in range(500)]).astype("float32"))
y = torch.from_numpy(np.asarray([[a + b] for a, b in zip(x, n)]))

linearRegression = LinearRegression()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(linearRegression.parameters(), lr=LR)
for i in range(EPOCH):
    y_prediction = linearRegression(x)
    loss = loss_function(y_prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(linearRegression.state_dict())
print("MSE:", loss_function(linearRegression(x), y).data.tolist())
y_prediction = [float(linearRegression.forward(i)) for i in x]
plt.scatter(x, y, label="Real")
plt.plot(x, y_prediction, label="Prediction", c="red")
plt.legend(loc=0)
plt.show()

# x = y + n
n = np.random.normal(loc=0, scale=1, size=500)
y = torch.from_numpy(np.asarray([[random.uniform(-10, 10)] for i in range(500)]).astype("float32"))
x = torch.from_numpy(np.asarray([[a + b] for a, b in zip(y, n)]))
linearRegression = LinearRegression()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(linearRegression.parameters(), lr=LR)
for i in range(EPOCH):
    y_prediction = linearRegression(x)
    loss = loss_function(y_prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("MSE:", loss_function(linearRegression(x), y).data.tolist())
y_prediction = [float(linearRegression.forward(i)) for i in x]
plt.scatter(x, y, label="Real")
plt.plot(x, y_prediction, label="Prediction", c="red")
plt.legend(loc=0)
plt.show()
