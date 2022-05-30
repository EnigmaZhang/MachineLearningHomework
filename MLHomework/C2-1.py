import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sigma = 0.01
LR = 0.001
STEP = 1000
TEST_SIZE = 0.7


class Perception:
    def __init__(self, input_size):
        self.w = np.zeros(input_size)
        self.b = 1

    def output(self, input):
        return np.sign(np.matmul(input, self.w.T) + self.b)

    def step(self, train, times):
        for i in range(times):
            deltas = []
            for data in train:
                delta = data[len(self.w):] - self.output(data[:len(self.w)])
                deltas.append(delta)
                self.b += LR * delta
                self.w += LR * delta * data[:len(self.w)]
            if abs(sum(deltas)) < 10E-5:
                break


# Generate samples.
positive = np.random.multivariate_normal(mean=(1, 1), cov=sigma * np.identity(2), size=100)
negative = np.concatenate((np.random.multivariate_normal(mean=(0, 0), cov=sigma * np.identity(2), size=100),
                           np.random.multivariate_normal(mean=(0, 1), cov=sigma * np.identity(2), size=100),
                           np.random.multivariate_normal(mean=(1, 0), cov=sigma * np.identity(2), size=100)))
plt.scatter(positive[:, 0], positive[:, 1], label="Positive")
plt.scatter(negative[:, 0], negative[:, 1], label="Negative")
plt.legend(loc='upper right')
plt.title("Sample Distribution")
plt.show()

perception = Perception(2)
positive = [[x[0], x[1], 1] for x in positive]
negative = [[x[0], x[1], -1] for x in negative]
train, test = train_test_split(positive + negative, test_size=TEST_SIZE)
perception.step(train, STEP)
test_result = [perception.output(x[:2]) for x in test]
true_result = [x[2] for x in test]
print(classification_report(true_result, test_result))


