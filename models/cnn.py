import tensorflow as tf
import tensorflow.keras.layers as Layers
from utils.process import nn_test, nn_train

class CNN(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.models.Sequential()
    self.model.add(Layers.Conv2D(32, 3, padding='same', activation='relu'))
    self.model.add(Layers.MaxPool2D())
    self.model.add(Layers.Conv2D(64, 3, padding='same', activation='relu'))
    self.model.add(Layers.MaxPool2D())
    self.model.add(Layers.Conv2D(128, 3, padding='same', activation='relu'))
    self.model.add(Layers.Flatten())
    self.model.add(Layers.Dropout(0.5))
    self.model.add(Layers.Dense(10, activation='softmax'))

  def call(self, x):
    return self.model(x)


def normalize(x, y):
  x = tf.cast(x, tf.float32)
  x = x / 255.0
  return tf.expand_dims(x,-1), y

def train(model=None, tag='default'):
  if model == None:
    model = CNN()
  nn_train(model, preprocess=normalize, tag=tag, lr=0.01)
  return model

def test(model, tag='default'):
  nn_test(model, preprocess=normalize, tag=tag)