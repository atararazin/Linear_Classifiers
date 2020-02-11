import numpy as np
import sys
from random import randint
from numpy import linalg as LA
import random


class Perceptron():
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self._eta = 0.01
        self._epochs = 20

    def train(self):
        w = np.zeros([3, 8])
        for e in range(self._epochs):
            mistake = 0
            for i in range(len(self.x_train)):
                x, y = shuffle(self.x_train, self.y_train)
                y_hat = int(np.argmax(np.dot(w, x)))
                if y != y_hat:
                    mistake += 1
                    w[y, :] = w[y, :] + self._eta * x
                    w[y_hat, :] = w[y_hat, :] - self._eta * x
        return w, mistake


class SVM():
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self._epochs = 20
        self._lambda = 0.001
        self._eta = 0.01

    def train(self):
        w = np.zeros([3, 8])
        for e in range(self._epochs):
            mistake = 0
            for i in range(len(self.x_train)):
                #shuffle
                x, y = shuffle(self.x_train, self.y_train)
                #predict
                y_hat = int(np.argmax(np.dot(w, x)))
                #update
                if y != y_hat:
                    mistake += 1
                    w[y, :] = w[y, :] * (1 - self._eta * self._lambda) + self._eta * x
                    w[y_hat, :] = w[y_hat, :] * (1 - self._eta * self._lambda) - self._eta * x
        return w, mistake

class PassiveAggressive():
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self._epochs = 25

    def train(self):
        w = np.zeros([3, 8])
        for e in range(self._epochs):
            mistake = 0
            for i in range(len(self.x_train)):
                x, y = shuffle(self.x_train, self.y_train)
                # predict
                y_hat = int(np.argmax(np.dot(w, x)))
                # update
                if y != y_hat:
                    mistake += 1
                    loss = max(0, 1 - np.dot(w[y, :], x) + np.dot(w[y_hat, :], x))
                    tau = loss / (2 * np.power(LA.norm(x), 2))
                    w[y, :] = w[y, :] + tau * x
                    w[y_hat, :] = w[y_hat, :] - tau * x
        return w, mistake

def shuffle(x,y):
    p = randint(0,len(x)-1)
    return x[p],int(y[p])


def load_data():
    train_x = open(sys.argv[1], 'r')
    train_x = np.loadtxt(train_x.name, dtype=str, delimiter=",")
    replaceCharsForInts(train_x)
    train_x = train_x.astype(np.float)

    train_y = open(sys.argv[2], 'r')
    train_y = np.loadtxt(train_y.name, delimiter=",")

    test_x = open(sys.argv[3], 'r')
    test_x = np.loadtxt(test_x.name, dtype=str, delimiter=',')
    replaceCharsForInts(test_x)
    test_x = test_x.astype(np.float)

    test_y = open(sys.argv[4], 'r')
    test_y = np.loadtxt(test_y.name, dtype=int, delimiter=',')
    return train_x, train_y, test_x,test_y

def convertCharToNum(c):
    if c =='M':
        return 1.0
    elif c == 'F':
        return 2.0
    else:#'I'
        return 3.0

def replaceCharsForInts(l):
    for col in l:
        res = convertCharToNum(col[0])
        col[0] = np.float(res)

def test(test_x, test_y):
    miss = 0
    for i in range(len(test_x)):
        res = int(np.argmax(np.dot(w, test_x[i])))
        if res != test_y[i]:
            miss += 1
    return miss

train_x, train_y, test_x,test_y = load_data()
percep = Perceptron(train_x, train_y)
svm = SVM(train_x, train_y)
pa = PassiveAggressive(train_x, train_y)

algos = [percep, svm, pa]
for algo in algos:
    w, mistakes = algo.train()
    miss = test(test_x, test_y)
    print(algo.__class__.__name__, "accuracy:", "{0:.2f}%".format((1 - miss / len(test_x)) * 100))
