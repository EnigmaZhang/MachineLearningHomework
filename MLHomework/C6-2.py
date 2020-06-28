import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = []

with open(r"./data/watermelon3.0.txt") as f:
    for line in f.readlines():
        line = line.strip().split(" ")
        if line[0] != "Density":
            data.append(line)

data = np.asarray(data, dtype="float32")
x, y = np.split(data, (2, ), axis=1)
y = y.squeeze(1).astype("int")

classifier = svm.LinearSVC()
classifier.fit(x, y)
prediction_y = classifier.predict(x)
print("Linear:")
print(classification_report(y, prediction_y))
decision_function = classifier.decision_function(x)
support_vector_indices = np.where((2 * y - 1) * decision_function <= 1)[0]
support_vectors = x[support_vector_indices]
print(support_vectors)

classifier = svm.SVC(kernel="rbf", decision_function_shape="ovr")
classifier.fit(x, y)
prediction_y = classifier.predict(x)
print("Gauss:")
print(classification_report(y, prediction_y))
print(classifier.support_vectors_)

