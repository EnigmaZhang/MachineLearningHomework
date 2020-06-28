from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
import collections
import torch

LR = 0.01
EPOCH = 500


class KNN:
    def __init__(self, x, y, k):
        self.train_x = x
        self.train_y = y
        self.k = k

    def fit(self, test):
        prediction_y = []
        for x in test:
            # Calculation distances of new data x
            distances = [sqrt(np.sum(train_x - x) ** 2) for train_x in self.train_x]
            # Find the nearest
            nearest = np.argsort(distances)
            topK_y = [self.train_y[i] for i in nearest[:self.k]]
            votes = collections.Counter(topK_y)
            prediction_y.append(votes.most_common(1)[0][0])
        return np.asarray(prediction_y)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.net(x)


data = []

with open(r"./data/iris.data", mode="r") as f:
    for lines in f.readlines():
        data.append(lines.strip().split(","))

# A [""] as last
data = data[:-1]
trans_dict = {"Iris-setosa": "0", "Iris-versicolor": "1", "Iris-virginica": "2"}
for i in data:
    i[-1] = trans_dict[i[-1]]

data = np.asarray(data, dtype="float32")
x, y = np.split(data, (4, ), axis=1)
y = y.squeeze(1)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

train_y_0 = train_y.copy()
for index, i in enumerate(train_y_0):
    if i < 0.9:
        train_y_0[index] = np.float32(1.)
    else:
        train_y_0[index] = np.float32(0.)
train_y_1 = train_y.copy()
for index, i in enumerate(train_y_1):
    if 0.5 < i < 1.5:
        train_y_1[index] = np.float32(1.)
    else:
        train_y_1[index] = np.float32(0.)
train_y_2 = train_y.copy()
for index, i in enumerate(train_y_2):
    if i > 1.9:
        train_y_2[index] = np.float32(1.)
    else:
        train_y_2[index] = np.float32(0.)

train_ys = [train_y_0, train_y_1, train_y_2]
final_prediction_y = np.zeros(test_x.shape[0])
final_prediction_y -= 1
for i, train_y_num in enumerate(train_ys):
    classifier = svm.SVC(kernel="linear")
    classifier.fit(train_x, train_y_num)
    prediction_y = classifier.predict(test_x)
    for index, y in enumerate(prediction_y):
        if y > 0.5:
            # Less samples have lower weight
            if final_prediction_y[index] < 0:
                final_prediction_y[index] = np.float32(i)
            else:
                if len(train_y_num) > len(train_ys[int(final_prediction_y[index])]):
                    final_prediction_y[index] = np.float32(i)
print(final_prediction_y)
print("My own SVM one vs rest:")
print(classification_report(test_y, final_prediction_y))

classifier = svm.SVC(kernel="linear", decision_function_shape="ovr")
classifier.fit(train_x, train_y)
prediction_y = classifier.predict(test_x)
print("Sklearn SVM one vs rest:")
print(classification_report(test_y, prediction_y))

my_knn = KNN(train_x, train_y, k=3)
prediction_y = my_knn.fit(test_x)
print("MyKNN:")
print(classification_report(test_y, prediction_y))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, train_y)
prediction_y = knn.predict(test_x)
print("KNN:")
print(classification_report(test_y, prediction_y))


train_ys = [train_y_0, train_y_1, train_y_2]
final_prediction_y = np.zeros(test_x.shape[0])
final_prediction_y -= 1
for i, train_y_num in enumerate(train_ys):
    classifier = MLP()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    for j in range(EPOCH):
        prediction_y = classifier.forward(torch.from_numpy(train_x).float())
        loss = loss_function(prediction_y, torch.from_numpy(train_y_num).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    prediction_y = [np.argmax(i) for i in classifier.forward(torch.from_numpy(test_x)).detach().numpy()]
    for index, y in enumerate(prediction_y):
        if y > 0.5:
            # Less samples have lower weight
            if final_prediction_y[index] < 0:
                final_prediction_y[index] = np.float32(i)
            else:
                if len(train_y_num) > len(train_ys[int(final_prediction_y[index])]):
                    final_prediction_y[index] = np.float32(i)
print("MLP: ")
print(classification_report(test_y, final_prediction_y))
