import tensorflow as tf

def create_padding_mask(decoder_token_ids):

    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    return seq[:, tf.newaxis, :]

def create_look_ahead_mask(sequence_length):

    #  Copy a tensor setting everything outside a central band in each innermost matrix to zero.
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask