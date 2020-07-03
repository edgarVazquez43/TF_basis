import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow import keras

print(tf.__version__)

###### CLASS FOR EARLY STOP
class new_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy')> 0.90): # select the accuracy
            print("\n !!! 90% accuracy, no further training !!!")
            self.model.stop_training = True


stop_callback = new_callback()
            
### Load the dataset
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

### Normalizing and Reshaping
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

### Defining the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

### Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Entrenamiento
model.fit(training_images, training_labels, epochs=10, callbacks = [stop_callback])

print("The training step has already finished....")

### Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
