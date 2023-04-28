from attention import Multihead_Attention
from fnn import FNNLayer
from tensorflow.keras.layers import Dropout, LayerNormalization
import tensorflow as tf

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, H, d_model, dk, dv, dff, dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()
        
        self.mha = Multihead_Attention(H, d_model, dk, dv)
        self.ffn = FNNLayer(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_mha = Dropout(dropout_rate)
        self.dropout_ffn = Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        A = self.mha(x,x,x,mask=mask)
        A = self.dropout_mha(A, training=training)
        
        out1 = self.layernorm1(x+A)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_ffn(ffn_output, training=training)

        encoder_layer_out = self.layernorm2(ffn_output+out1)
        
        return encoder_layer_out

class Encoder(tf.keras.layers.Layer):

    def __init__(self, N, H, d_model, dk, dv, dff, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()
        
        self.layers=[EncoderLayer(H, d_model, dk, dv, dff, 
                                  dropout_rate=dropout_rate, 
                                  layernorm_eps=layernorm_eps)
                                  for i in range(N)]
    
    def call(self, x, training=False, mask=None):
        for layer in self.layers:
            x = layer(x, training=training, mask=mask)
                                  
        return x