import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow import keras

print(tf.__version__)


mnist = tf.keras.datasets.fashion_mnist



### Load the dataset
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


### Normalizing
training_images  = training_images / 255.0
test_images = test_images / 255.0

### Defining the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])


### Compiling the model
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

### Entrenamiento
model.fit(training_images, training_labels, epochs = 50)


print("The training step has already finished....")


### Evaluation
model.evaluate(test_images, test_labels)
