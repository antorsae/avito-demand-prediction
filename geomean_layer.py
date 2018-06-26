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
            initializer='ones',
            trainable=True)
        super(GeomeanLayer, self).build(input_shape)

    def call(self, x):
        tf = K.tf
        t = tf.exp(tf.reduce_sum(self.regression_weights*tf.log(tf.clip_by_value(x, K.epsilon(), 1.0)), 1)/ tf.reduce_sum(self.regression_weights+K.epsilon()))
        t = tf.reshape(t, [-1, 1])
        print(t.get_shape())
        return t

        #return tf.reshape(tf.reduce_sum(self.regression_weights*x, 1) / tf.reduce_sum(self.regression_weights+K.epsilon()), [-1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
