from tensorflow.keras.datasets import mnist
import model
import numpy as np
import matplotlib.pyplot as plt

# get data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# model
model = model.create_model()
model.load_weights("model/trained_model.ckpt")

# eval
model.evaluate(x_test, y_test)

# show false predictions
predictions = np.argmax(model.predict(x_test), axis=1)
for i, prediction in enumerate(predictions):
    if prediction != y_test[i]:
        plt.title("Expected: {}, Actual: {}".format(y_test[i], prediction))
        plt.imshow(x_test[i], cmap="gray")
        plt.show()
