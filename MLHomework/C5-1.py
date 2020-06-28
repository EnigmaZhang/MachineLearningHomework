import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import torch
import matplotlib.pyplot as plt

LR = 0.01
EPOCH = 1000


class MLP_1(torch.nn.Module):
    def __init__(self):
        super(MLP_1, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


class MLP_2(torch.nn.Module):
    def __init__(self):
        super(MLP_2, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)


positive_x = np.random.multivariate_normal(mean=[2, 3], cov=np.identity(2), size=200)
negative_x = np.random.multivariate_normal(mean=[5, 6], cov=np.identity(2), size=800)

positive_y = np.asarray([1] * 200)
negative_y = np.asarray([0] * 800)
x, test_x, y, test_y = train_test_split(np.concatenate([positive_x, negative_x], axis=0),
                                        np.concatenate([positive_y, negative_y], axis=0), test_size=0.2)
train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=200)
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).long()
validation_x = torch.from_numpy(validation_x).float()
validation_y = torch.from_numpy(validation_y).long()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).long()

mlp_1 = MLP_1()
mlp_2 = MLP_2()
loss_function = torch.nn.CrossEntropyLoss()
optimizer_1 = torch.optim.Adam(mlp_1.parameters(), lr=LR)
optimizer_2 = torch.optim.Adam(mlp_2.parameters(), lr=LR)

# train_losses = []
# cross_losses = []

# Learning curve MLP 1
# for split_range in range(1, 600, 50):
#     cross_train_x, cross_x, cross_train_y, cross_y = train_test_split(train_x, train_y, test_size=600 - split_range)
#     for i in range(EPOCH):
#         y_prediction = mlp_1.forward(cross_train_x)
#         loss = loss_function(y_prediction, cross_train_y)
#         optimizer_1.zero_grad()
#         loss.backward()
#         optimizer_1.step()
#     train_losses.append(loss_function(mlp_1.forward(cross_train_x), cross_train_y))
#     cross_losses.append(loss_function(mlp_1.forward(cross_x), cross_y))

# Learning curve MLP 2
# for split_range in range(1, 600, 50):
#     cross_train_x, cross_x, cross_train_y, cross_y = train_test_split(train_x, train_y, test_size=600 - split_range)
#     for i in range(EPOCH):
#         y_prediction = mlp_2.forward(cross_train_x)
#         loss = loss_function(y_prediction, cross_train_y)
#         optimizer_2.zero_grad()
#         loss.backward()
#         optimizer_2.step()
#     train_losses.append(loss_function(mlp_2.forward(cross_train_x), cross_train_y))
#     cross_losses.append(loss_function(mlp_2.forward(cross_x), cross_y))

# Plot Learning curve
# plt.ylim(0, 0.5)
# plt.plot(train_losses, label="Train")
# plt.plot(cross_losses, label="Validation")
# plt.title("Learning Curve")
# plt.legend(loc=0)
# plt.show()

# Train
losses = []
for i in range(EPOCH):
    y_prediction = mlp_1.forward(train_x)
    loss = loss_function(y_prediction, train_y)
    losses.append(loss)
    optimizer_1.zero_grad()
    loss.backward()
    optimizer_1.step()

# plt.plot(losses)
# plt.show()

# validation_prediction = [np.argmax(i) for i in mlp_1.forward(validation_x).detach().numpy()]
# print(classification_report(validation_y.detach().numpy(), validation_prediction))

test_prediction = [np.argmax(i) for i in mlp_1.forward(test_x).detach().numpy()]
print(classification_report(test_y.detach().numpy(), test_prediction))

fpr_1, tpr_1, thresholds_1 = roc_curve(test_y.detach().numpy(), test_prediction)

# Train
losses = []
for i in range(EPOCH):
    y_prediction = mlp_2.forward(train_x)
    loss = loss_function(y_prediction, train_y)
    losses.append(loss)
    optimizer_2.zero_grad()
    loss.backward()
    optimizer_2.step()

# plt.plot(losses)
# plt.show()

# validation_prediction = [np.argmax(i) for i in mlp_2.forward(validation_x).detach().numpy()]
# print(classification_report(validation_y.detach().numpy(), validation_prediction))
test_prediction = [np.argmax(i) for i in mlp_2.forward(test_x).detach().numpy()]
print(classification_report(test_y.detach().numpy(), test_prediction))

fpr_2, tpr_2, thresholds_2 = roc_curve(test_y.detach().numpy(), test_prediction)
plt.plot(fpr_1, tpr_1, label="MLP_1")
plt.plot(fpr_2, tpr_2, label="MLP_2")
plt.legend(loc=0)
plt.show()

print("AUC 1: {}".format(auc(fpr_1, tpr_1)))
print("AUC 2: {}".format(auc(fpr_2, tpr_2)))
