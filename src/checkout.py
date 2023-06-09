import tensorflow as tf
import tensorboard
import numpy
import keras
import datetime
import config_file
import os
import gym

# Check avalible deivces
print(tf.config.list_physical_devices('GPU'))

print(tf.__version__)

print(tensorboard.__version__)

print(numpy.__version__)

print(keras.__version__)

print(gym.__version__) #type: ignore

print(datetime.datetime.now())

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense1'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense2'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense3'),
    tf.keras.layers.Dropout(0.2, name='layers_dropout'),
    tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
  ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config_file.LOG_DIR, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=100, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

# save model
model.save(config_file.MODELS_DIR + 'checkout_model.h5')
