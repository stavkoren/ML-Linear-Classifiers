import random
import sys
import numpy as np


def shuffle(a, b):
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]


def getSecondMaxValue(arr):
    idx = max(np.where(arr == np.amax(np.partition(arr.flatten(), -2)[-2])))[0]
    return idx


def printPredictions(perceptron, svm, pa, predict_x):
    for x in predict_x:
        print(f"perceptron: {perceptron.predict(x)}, svm: {svm.predict(x)}, pa: {pa.predict(x)}")


class Perceptron:
    epochs = 200
    weight = []
    eta = 0.85
    optimal_epochs = 0
    best_success_rate = 0

    def train(self, train_x, train_y, predict_x, predict_y):
        w = [[0.0 for i in range(len(train_x[0]))] for j in range(len(dic))]
        w = np.array(w)
        self.weight = [[0.0 for i in range(len(train_x[0]))] for j in range(len(dic))]
        self.weight = np.array(w)
        for e in range(self.epochs):
            train_x, train_y = shuffle(train_x, train_y)
            for x, y in zip(train_x, train_y):
                # predict
                y_hat = np.argmax(np.dot(w, x))
                # update
                if y != y_hat:
                    w[y, :] += self.eta * x
                    w[y_hat, :] -= self.eta * x
            current_succeess_rate = self.successRate(predict_x, predict_y)
            if self.best_success_rate < current_succeess_rate:
                self.best_success_rate = current_succeess_rate
                self.weight = w
                self.optimal_epochs = e

    def predict(self, x):
        return np.argmax(np.dot(self.weight, x))

    def successRate(self, predict_x, predict_y):
        count_success = 0
        for x, y in zip(predict_x, predict_y):
            y_hat_pa = self.predict(x)
            if y == y_hat_pa:
                count_success += 1
        success_rate = count_success / (len(predict_y))
        return success_rate


class PA:
    max_epochs = 200
    weight = []
    optimal_epochs = 0
    best_success_rate = 0

    def train(self, train_x, train_y, predict_x, predict_y):
        w = [[0.0 for i in range(len(train_x[0]))] for j in range(len(dic))]
        w = np.array(w)
        self.weight = [[0.0 for i in range(len(train_x[0]))] for j in range(len(dic))]
        self.weight = np.array(self.weight)
        for e in range(self.max_epochs):
            train_x, train_y = shuffle(train_x, train_y)
            for x, y in zip(train_x, train_y):
                # predict
                w_mul_x = np.dot(w, x)
                y_hat = np.argmax(w_mul_x)
                if self.loss(x, y, w, y_hat) > 0:
                    # update
                    if y == y_hat:
                        y_hat = getSecondMaxValue(w_mul_x)
                    tau = self.tau(x, y, w, y_hat)
                    w[y, :] += tau * x
                    w[y_hat, :] -= tau * x
            current_succeess_rate = self.successRatePA(predict_x, predict_y)
            if self.best_success_rate < current_succeess_rate:
                self.best_success_rate = current_succeess_rate
                self.weight = w
                self.optimal_epochs = e

    def loss(self, x, y, w, y_hat):
        return max(0, (1 - (np.dot(w[y, :], x)) + (np.dot(w[y_hat, :], x))))

    def tau(self, x, y, w, t_hat):
        return (self.loss(x, y, w, t_hat)) / (2 * (np.power(np.linalg.norm(x), 2)))

    def predict(self, x):
        return np.argmax(np.dot(self.weight, x))

    def successRatePA(self, predict_x, predict_y):
        count_success_pa = 0
        for x, y in zip(predict_x, predict_y):
            y_hat_pa = self.predict(x)
            if y == y_hat_pa:
                count_success_pa += 1
        success_rate = count_success_pa / (len(predict_y))
        return success_rate


class SVM:
    epochs = 200
    weight = []
    eta = 0.099
    lamda = 0.11
    optimal_epochs = 0
    best_success_rate = 0

    def train(self, train_x, train_y, predict_x, predict_y):
        self.weight = [[0.0 for i in range(len(train_x[0]))] for j in range(len(dic))]
        self.weight = np.array(self.weight)
        w = [[0.0 for i in range(len(train_x[0]))] for j in range(len(dic))]
        w = np.array(w)
        for e in range(self.epochs):
            train_x, train_y = shuffle(train_x, train_y)
            for x, y in zip(train_x, train_y):
                # predict
                w_mul_x = np.dot(w, x)
                y_hat = np.argmax(w_mul_x)
                if self.loss(x, y, w, y_hat) > 0:
                    # update
                    if y == y_hat:
                        y_hat = getSecondMaxValue(w_mul_x)
                    w[y, :] = np.dot((1 - self.eta * self.lamda), w[y, :]) + self.eta * x
                    w[y_hat, :] = np.dot((1 - self.eta * self.lamda), w[y_hat, :]) - self.eta * x
                    for i in range(w.shape[0]):
                        if i != y_hat and i != y:
                            w[i, :] *= (1 - self.eta * self.lamda)
            current_succeess_rate = self.successRateSVM(predict_x, predict_y)
            if self.best_success_rate < current_succeess_rate:
                self.best_success_rate = current_succeess_rate
                self.weight = w
                self.optimal_epochs = e

    def predict(self, x):
        return np.argmax(np.dot(self.weight, x))

    def loss(self, x, y, w, y_hat):
        return max(0, (1 - (np.dot(w[y, :], x)) + (np.dot(w[y_hat, :], x))))

    def successRateSVM(self, predict_x, predict_y):
        count_success_svm = 0
        for x, y in zip(predict_x, predict_y):
            y_hat_svm = self.predict(x)
            if y == y_hat_svm:
                count_success_svm += 1
        success_rate = count_success_svm / (len(predict_y))
        return success_rate


train_x, train_y, predict_x = sys.argv[1], sys.argv[2], sys.argv[3]
dic = {
    "M": 1,
    "F": 2,
    "I": 3
}
# load train set
train_x = np.genfromtxt(train_x, delimiter=",", dtype="str")
train_y = np.loadtxt(train_y, dtype=np.int32)
predict_x = np.genfromtxt(predict_x, delimiter=",", dtype="str")
# replace characters with numbers
for row in train_x:
    row[0] = dic[row[0]]
for row in predict_x:
    row[0] = dic[row[0]]
# convert to numpy of floats
train_x = train_x.astype(float)
predict_x = predict_x.astype(float)
svm = SVM()
svm.train(train_x[:int(len(train_y) * 0.8), :], train_y[:int(len(train_y) * 0.8)],
          train_x[int(len(train_y) * 0.8):, :], train_y[int(len(train_y) * 0.8):])
# print("success rate SVM:", svm.best_success_rate * 100, "ephocs:", svm.optimal_epochs)
perceptron = Perceptron()
perceptron.train(train_x[:int(len(train_y) * 0.8), :], train_y[:int(len(train_y) * 0.8)],
                 train_x[int(len(train_y) * 0.8):, :], train_y[int(len(train_y) * 0.8):])
# print("success rate Perceptron:", perceptron.best_success_rate * 100,"ephocs:", perceptron.optimal_epochs)
pa = PA()
pa.train(train_x[:int(len(train_y) * 0.8), :], train_y[:int(len(train_y) * 0.8)],
         train_x[int(len(train_y) * 0.8):, :], train_y[int(len(train_y) * 0.8):])
# print("success rate PA:", pa.best_success_rate * 100, "ephocs:", pa.optimal_epochs)

printPredictions(perceptron, svm, pa, predict_x)
