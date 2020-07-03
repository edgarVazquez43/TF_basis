import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

dataset = tf.data.Dataset.range(10)
print("Simple dataset array: ")
for val in dataset:
   print(val.numpy())

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
print("Data windowed by 5:")
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()


dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
print("Data windowed by 5, shift 1, drop_remainder:")
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
print("Flat-map with lambda fuction: ")
for window in dataset:
  print(window.numpy())


dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
print("Flat-map with lambda fuction, split data X-Y: ")
for x,y in dataset:
  print(x.numpy(), y.numpy())

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
print("After shuffle: ")
for x,y in dataset:
  print(x.numpy(), y.numpy())


dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
print("After batching by 2: ")
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
