from attention import Multihead_Attention
from fnn import FNNLayer
from tensorflow.keras.layers import Dropout, LayerNormalization
import tensorflow as tf

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, H, d_model, dk, dv, dff, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = Multihead_Attention(H, d_model, dk, dv)
        self.mha2 = Multihead_Attention(H, d_model, dk, dv)
        self.ffn = FNNLayer(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_mha1 = Dropout(dropout_rate)
        self.dropout_mha2 = Dropout(dropout_rate)                                     
        self.dropout_ffn = Dropout(dropout_rate)
    
    def call(self, x, encoder_output, training=False, look_ahead_mask=None, padding_mask=None):
        A1 = self.mha1(x,x,x,mask=look_ahead_mask)
        A1 = self.dropout_mha1(A1, training=training)

        out1 = self.layernorm1(x+A1)
    
        A2 = self.mha2(x,encoder_output,encoder_output,mask=padding_mask)
        A2 = self.dropout_mha2(A2, training=training)

        out2 = self.layernorm2(out1+A2)
                                             
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_ffn(ffn_output, training=training)

        decoder_layer_out = self.layernorm3(ffn_output+out2)
        
        return decoder_layer_out

class Decoder(tf.keras.layers.Layer):

    def __init__(self, N, H, d_model, dk, dv, dff, dropout_rate=0.1, layernorm_eps=1e-6):

        super(Decoder, self).__init__()
        
        self.layers=[DecoderLayer(H, d_model, dk, dv, dff, 
                                  dropout_rate=dropout_rate, 
                                  layernorm_eps=layernorm_eps)
                                  for i in range(N)]
    
    def call(self, x, encoder_output, training=False, look_ahead_mask=None, padding_mask=None):
        for layer in self.layers:
            x = layer(x,encoder_output, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
                                  
        return x