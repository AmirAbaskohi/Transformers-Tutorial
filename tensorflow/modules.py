import numpy as np
import tensorflow as tf
import math

def scaled_dot_product_attention(Q, K, V, key_masks,
                                causality=False, dropout_rate=0.,
                                training=True,
                                scope="scaled_dot_product_attention"):
    
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs /= math.sqrt(d_k)

        outputs = mask(outputs, key_masks=key_masks, type="key")

        if causality:
            outputs = mask(outputs, type="future")

        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        outputs = tf.matmul(outputs, V)

    return outputs

