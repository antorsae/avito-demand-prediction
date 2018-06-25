from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class GeomeanLayer(Layer):
    def __init__(self, **kwargs):
        super(GeomeanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.regression_weights = self.add_weight(
            name='regression_weights',
            shape=(input_shape[1], ),
            initializer='uniform',
            trainable=True)
        super(GeomeanLayer, self).build(input_shape)

    def call(self, x):
        tf = K.tf
        #return tf.pow(tf.reduce_prod(tf.pow(x, self.regression_weights), axis=-1), 1.0/tf.reduce_sum(self.regression_weights))
        return tf.reduce_sum(self.regression_weights*x, 1) / tf.reduce_sum(self.regression_weights+K.epsilon())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
