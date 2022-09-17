import numpy as np
import tensorflow as tf
import math

def scaled_dot_product_attention(Q, K, V, key_masks,
                                causality=False,
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

        outputs = tf.layers.dropout(outputs, training=training)

        outputs = tf.matmul(outputs, V)

    return outputs

def multihead_attention(q, k, v, key_masks, n_heads, training=True, causality=False, scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(q, d_model, use_bias=True)
        K = tf.layers.dense(k, d_model, use_bias=True)
        V = tf.layers.dense(v, d_model, use_bias=True)

        Q = tf.concat(tf.split(Q, n_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, n_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, n_heads, axis=2), axis=0)

        outputs = scaled_dot_product_attention(Q, K, V, key_masks, causality, training)

        outputs = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2)

        return outputs