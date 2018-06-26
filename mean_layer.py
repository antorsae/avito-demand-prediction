from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.initializers import Constant

class MeanLayer(Layer):
    def __init__(self, **kwargs):
        super(MeanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.regression_weights = self.add_weight(
            name='regression_weights',
            shape=(input_shape[1], ),
            initializer='uniform',
            trainable=True)
        self.regression_bias = self.add_weight(
            name='regression_bias',
            shape=(1, ),
            initializer='zero',
            trainable=True)
        super(MeanLayer, self).build(input_shape)

    def call(self, x):
        tf = K.tf
        return tf.reshape(
            tf.reduce_sum(self.regression_weights * x, 1) /
            tf.reduce_sum(self.regression_weights + K.epsilon()),
            [-1, 1]) + self.regression_bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
