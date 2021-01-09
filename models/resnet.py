import tensorflow as tf
import tensorflow.keras.layers as Layers
from utils.process import nn_test, nn_train

class ResBlock(Layers.Layer):
  def __init__(self, fn, k):
    super().__init__()
    self.conv1 = Layers.Conv2D(filters=fn, kernel_size=k, padding='same', activation='relu')
    self.conv2 = Layers.Conv2D(filters=fn, kernel_size=k, padding='same')
    self.relu = Layers.ReLU()
  
  def call(self, x):
    r = self.conv1(x)
    r = self.conv2(r)
    return self.relu(x + r)

class ResNet(tf.keras.Model):
  def __init__(self):
    super().__init__()
    resnet = tf.keras.Sequential()
    resnet.add(Layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(Layers.MaxPool2D())
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(Layers.MaxPool2D())
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(ResBlock(fn=16, k=3))
    resnet.add(Layers.Flatten())
    resnet.add(Layers.Dropout(0.5))
    resnet.add(Layers.Dense(10, activation='softmax'))
    self.resnet = resnet
  
  def call(self, x):
    return self.resnet(x)

def normalize(x, y):
  x = tf.cast(x, tf.float32)
  x = x / 255.0
  return tf.expand_dims(x,-1), y

def train(model=None, tag='default'):
  if model == None:
    model = ResNet()
  nn_train(model, preprocess=normalize, tag=tag, lr=0.01)
  return model

def test(model, tag='default'):
  nn_test(model, preprocess=normalize, tag=tag)