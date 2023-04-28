import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_QK = tf.matmul(Q,K,transpose_b=True)

    dk = K.shape[-1]
    scaled_attention_logits = matmul_QK/np.sqrt(dk)

    if mask is not None: 
        scaled_attention_logits += (1. - mask) *(-1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights,V)
    
    return output, attention_weights

class Multihead_Attention(tf.keras.layers.Layer):
    def __init__(self, H, d_model, dk, dv):    
        super(Multihead_Attention, self).__init__()
        
        initializer = tf.keras.initializers.GlorotUniform()
        self.WQ = tf.Variable(initializer(shape=(H, d_model, dk)), trainable=True)
        self.WK = tf.Variable(initializer(shape=(H, d_model, dk)), trainable=True)
        self.WV = tf.Variable(initializer(shape=(H, d_model, dv)), trainable=True)
        self.WO = tf.Variable(initializer(shape=(H*dv,d_model)), trainable=True)

    
    def call(self, Q, K, V, mask=None):
        Qh= tf.experimental.numpy.dot(Q, self.WQ)
        Kh= tf.experimental.numpy.dot(K, self.WK)
        Vh= tf.experimental.numpy.dot(V, self.WV)
        
        Qh=tf.transpose(Qh, [0,2,1,3])
        Kh=tf.transpose(Kh, [0,2,1,3])
        Vh=tf.transpose(Vh, [0,2,1,3])

        Ah,_=scaled_dot_product_attention(Qh, Kh, Vh, mask=mask)
        
        s=Ah.shape
        A = tf.reshape(Ah,(s[0],s[2],s[1]*s[3]))
        A= tf.experimental.numpy.dot(A, self.WO)
        
        return A