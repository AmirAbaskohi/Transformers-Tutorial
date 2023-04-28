import tensorflow as tf
from tensorflow.keras.layers import Conv1D

class FNNLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, dff):

        super(FNNLayer, self).__init__()

        self.layer1 = Conv1D(filters=dff, kernel_size=1, activation='relu')
        self.layer2 = Conv1D(filters=d_model, kernel_size=1)

    def call(self, x):

        x = self.layer1(x)
        fnn_layer_out = self.layer2(x)

        return fnn_layer_out