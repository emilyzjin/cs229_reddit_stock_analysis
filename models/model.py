import numpy as np
from numpy import linalg
from utils.util import data_iter


class linearRegression:
    def __init__(self, learning_rate, num_epochs, batch_size=32, theta=None, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.theta = theta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss = None
        self.epsilon = epsilon

    def train(self, x_train, y_train, verbose=False):
        x_train = np.insert(x_train, 0, 1, axis=1)
        x_train.setflags(write=False)

        theta = np.zeros(x_train.shape[1])

        for epoch in range(self.num_epochs):
            for x, y in data_iter(self.batch_size, x_train, y_train):
                grad = np.mean((y - x.dot(theta))[:, np.newaxis] * x, axis=0)
                update = self.learning_rate * grad
                theta += update
                loss = np.sum(np.square(x_train.dot(theta) - y_train)) / 2
                if np.isnan(loss):
                    self.theta = theta
            if verbose:
                print(f'epoch {epoch + 1}, loss {float(loss.mean()):f}')
            if np.linalg.norm(update) <= self.epsilon:
                break

        self.theta = theta
        self.loss = loss

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        x.setflags(write=False)
        preds = x.dot(self.theta)

        return preds

    def evaluate(self, x, y):
        y_hat = self.predict(x)
        total_deviation = np.mean(y_hat - y)
        total_magnitude_deviation = np.mean(np.abs(y_hat - np.abs(y)))
        rmse = np.sqrt(np.mean(np.square(y-y_hat)))

        results = {'Model': self,
                   'lr': self.learning_rate,
                   'Dev': total_deviation,
                   'Mag_Dev': total_magnitude_deviation,
                   'RMSE': rmse,
                   'Preds': y_hat
                   }
        return results

