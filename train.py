import tensorflow as tf
from tensorflow.keras.datasets import mnist
import model

# get data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalization
x_train = x_train / 255.0

# set save path
save_path = "model/trained_model.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# build model
model = model.create_model()

# training
model.fit(x_train, y_train, epochs=5,
          callbacks=[cp_callback])



