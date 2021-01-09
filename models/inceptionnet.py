import tensorflow as tf
import tensorflow.keras.layers as Layers
from utils.process import nn_train, nn_test

def make_conv(fn, ks):
  return Layers.Conv2D(fn, ks, padding='same', activation='relu')

class InceptionBlock(tf.keras.layers.Layer):
  def __init__(self, fn1, rn3, fn3, rn5, fn5, pn):
    super().__init__()
    self.br_1x1 = make_conv(fn1, 1)
    self.br_3x3 = tf.keras.Sequential([make_conv(rn3, 1), make_conv(fn3, 3)])
    self.br_5x5 = tf.keras.Sequential([make_conv(rn5, 1), make_conv(fn5, 3)])
    self.br_pool = tf.keras.Sequential([Layers.MaxPool2D((3,3), 1, 'same'), make_conv(pn,1)])

  def call(self, x):
    return tf.concat([self.br_1x1(x), self.br_3x3(x), self.br_5x5(x), self.br_pool(x)], -1)


class InceptionNet(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.Sequential()
    self.model.add(make_conv(32, 3))
    self.model.add(Layers.MaxPool2D(padding='same'))
    self.model.add(InceptionBlock(20, 8, 20, 6, 12, 12))
    self.model.add(InceptionBlock(20, 8, 20, 6, 12, 12))
    self.model.add(Layers.MaxPool2D(padding='same'))
    self.model.add(InceptionBlock(36, 12, 36, 10, 28, 28))
    self.model.add(InceptionBlock(36, 12, 36, 10, 28, 28))
    self.model.add(InceptionBlock(36, 12, 36, 10, 28, 28))
    self.model.add(Layers.AvgPool2D(padding='same'))
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
    model = InceptionNet()
  nn_train(model, preprocess=normalize, tag=tag, lr=0.01)
  return model

def test(model, tag='default'):
  nn_test(model, preprocess=normalize, tag=tag)