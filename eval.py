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

# show 25 first predictions
predictions = np.argmax(model.predict(x_test), axis=1)
plt.figure(figsize=(10, 10))
for i in range(25):
    img = x_test[i]
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap="gray")
    plt.xlabel("E: {}, A: {}".format(y_test[i], predictions[i]))
plt.show()