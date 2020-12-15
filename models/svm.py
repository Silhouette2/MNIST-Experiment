
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# from settings import *

from utils.loggers import test_logger, train_logger
import tensorflow as tf
import numpy as np

def __getProjection(img):
  def project(x, axis):
    return tf.reduce_sum(tf.cast(tf.reduce_sum(x, axis) > 0, tf.uint8))

  return [project(img[  : 7,  :14], 0), project(img[  : 7,14:  ], 0),
          project(img[ 7:14,  :14], 0), project(img[ 7:14,14:  ], 0),
          project(img[14:21,  :14], 0), project(img[14:21,14:  ], 0),
          project(img[21:  ,  :14], 0), project(img[21:  ,14:  ], 0),
          project(img[  :14,  : 7], 1), project(img[14:  ,  : 7], 1),
          project(img[  :14, 7:14], 1), project(img[14:  , 7:14], 1),
          project(img[  :14,14:21], 1), project(img[14:  ,14:21], 1),
          project(img[  :14,21:  ], 1), project(img[14:  ,21:  ], 1)]

def __getDistribution(img):
  return [tf.reduce_sum(tf.cast(img[0:9, 0:9] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[0:9, 9:19] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[0:9, 19:28] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[9:19, 0:9] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[9:19, 9:19] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[9:19, 19:28] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[19:28, 0:9] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[19:28, 9:19] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[19:28, 19:28] > 0, tf.uint8))]


def __transform(x, y):
  f_proj = __getProjection(x)
  f_dist = __getDistribution(x)
  return tf.convert_to_tensor(f_proj + f_dist), y


def train(model=None, name='default', **model_args):
  train_val_data, _ = tf.keras.datasets.mnist.load_data()
  train_val_ds = tf.data.Dataset.from_tensor_slices(train_val_data).shuffle(60000)
  train_ds = train_val_ds.take(10000).map(__transform).batch(10000)
  val_ds = train_val_ds.skip(10000).take(5000).map(__transform).batch(5000)

  if model == None:
    model = make_pipeline(StandardScaler(), SVC(**model_args))
  
  X_train, Y_train = next(iter(train_ds))
  model.fit(X_train, Y_train)

  X_val, Y_val = next(iter(val_ds))
  P_val = model.predict(X_val)
  train_logger.critical('[{:s}] Validation Accuracy: {:.3%}'.format(name, (P_val == Y_val.numpy()).sum() / P_val.shape[0]))

  return model


def test(model, name='default'):
  _, test_data = tf.keras.datasets.mnist.load_data()
  test_ds = tf.data.Dataset.from_tensor_slices(test_data).map(__transform).batch(10000)
  
  X_test, Y_test = next(iter(test_ds))
  P_test = model.predict(X_test)
  test_logger.critical('[{:s}] Test Accuracy: {:.3%}'.format(name, (P_test == Y_test.numpy()).sum() / P_test.shape[0]))
