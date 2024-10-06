import numpy as np
import copy
import math

class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        limit = 1 / math.sqrt(input_shape[0])
        self.W = np.random.uniform(-limit, limit, size=(input_shape[0], n_units))
        self.w0 = 0

    def initialize(self, optimizer):
          self.W_optimizer = copy.copy(optimizer)
          self.b_optimizer = copy.copy(optimizer)

    def forward_pass(self, X):
        self.layer_input = X
        return np.dot(X, self.W) + self.w0
    
    def backward_pass(self, accum_grad):
        dW = self.layer_input.T.dot(accum_grad)
        db = np.sum(accum_grad, axis=0, keepdims=True)
        accum_grad = accum_grad.dot(self.W.T)

        self.W = self.W_optimizer.update(self.W, dW)
        self.w0 = self.b_optimizer.update(self.w0, db)

        return accum_grad

    def number_of_parameters():
        pass