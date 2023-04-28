import tensorflow as tf
from transfomer import Transformer

N, H, d_model, dk, dv, dff = 6, 8, 512, 64, 64, 2048
vocab_size, T =29, 11
batch_size = 3


transformer = Transformer(N, H, d_model, dk, dv, dff, vocab_size, T)

input_shape = (None, T,vocab_size)


x = tf.random.uniform((batch_size, T, vocab_size))
y = tf.random.uniform((batch_size, T, vocab_size))

pred = transformer(x,y,training=True)
print(pred.shape)

print(transformer.summary())

print("--------------------------------------------------")

warmup_step = 4000
class LearningRateScheduler(tf.keras.callbacks.Callback):
    def on_train_batch_start(self, i, batch_logs):
        transformer.optimizer.lr = dk**(-0.5)*min(i**(-0.5),warmup_step**(-3/2)*i)


callback = LearningRateScheduler()

optimizer = tf.keras.optimizers.Adam(learning_rate=0, beta_1=0.9, beta_2=0.98, epsilon=1e-09)

transformer.compile(loss='crossentropy',optimizer=optimizer,metrics=['accuracy'])

print("Model compiled")