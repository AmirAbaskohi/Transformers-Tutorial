from encoder import Encoder
from decoder import Decoder
from positional_encoding import positional_encoding
import tensorflow as tf
from tensorflow.keras.layers import Dropout, LayerNormalization

class Transformer(tf.keras.Model):
    
    def __init__(self, N, H, d_model, dk, dv, dff, 
                 vocab_size, max_positional_encoding, 
                 dropout_rate=0.1, layernorm_eps=1e-6):

        super(Transformer, self).__init__()
        
        initializer = tf.keras.initializers.GlorotUniform()
        self.embedding = tf.Variable(initializer(shape=(vocab_size, d_model)), trainable=True)
        self.PE = positional_encoding(max_positional_encoding, d_model)
        
        self.dropout_encoding_input = Dropout(dropout_rate)
        self.dropout_decoding_input = Dropout(dropout_rate)
        
        self.encoder = Encoder(N, H, d_model, dk, dv, dff, dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)
        self.decoder = Decoder(N, H, d_model, dk, dv, dff, dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)

        

    def call(self, x, y, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        
        x = tf.matmul(x,self.embedding)
        x = x + self.PE
        x =  self.dropout_encoding_input(x,training=training)
        
        encoder_output = self.encoder(x,training=training, mask=enc_padding_mask)
        
        y = tf.matmul(y,self.embedding)
        y = y + self.PE
        y = self.dropout_decoding_input(y,training=training)
        
        dec_output = self.decoder(y, encoder_output, training=training, 
                                  look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        
        
        pred =  tf.matmul(self.embedding,dec_output,transpose_b=True)
        pred = tf.nn.softmax(pred)
        
        return pred