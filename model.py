import tensorflow as tf


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # input
        tf.keras.layers.Dense(128, activation='relu'),  # hidden
        tf.keras.layers.Dense(10, activation='softmax')  # output
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
