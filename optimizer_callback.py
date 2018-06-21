from keras.callbacks import *
from keras.optimizers import SGD, Adam, Adadelta

class OptimizerCallback(Callback):
        def __init__(self, lr=0.01):
            super(OptimizerCallback, self).__init__()
            self.adam = Adam(lr=lr)
            self.sgd = SGD(lr=lr, momentum=0.9)

        def on_batch_begin(self, batch, logs=None):
            if batch % 2 == 0:
                self.model.optimizer = self.adam
            elif batch %2 == 1:
                self.model.optimizer = self.sgd
