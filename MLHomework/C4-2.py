import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LR = 0.01
EPOCH = 2000


class Regression(torch.nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 64),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


data = pd.read_csv(r"./data/forestfires.csv")
x = np.asarray(data[["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]].values)
scalar = StandardScaler()
x = scalar.fit_transform(x)
x = torch.from_numpy(x).float()
y = torch.from_numpy(np.asarray(data["area"].values).reshape(517, 1)).float()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


regression = Regression()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(regression.parameters(), lr=LR)
for i in range(EPOCH):
    y_prediction = regression.forward(x_train)
    loss = loss_function(y_prediction, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss_function(regression.forward(x_test), y_test))
