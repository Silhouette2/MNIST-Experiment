import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils.process import sklearn_train, sklearn_test

def flatten(x, y):
  return tf.reshape(x, [-1]), y

def train(model, tag='default'):
  if model == None:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11))
  sklearn_train(model, preprocess=flatten, tag=tag)
  return model

def test(model, tag='default'):
  sklearn_test(model, preprocess=flatten, tag=tag)