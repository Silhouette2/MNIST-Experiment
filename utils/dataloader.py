import tensorflow as tf
from settings import *

def load_full(preprocess):
  train_val_data, test_data = tf.keras.datasets.mnist.load_data()
  train_val_ds = tf.data.Dataset.from_tensor_slices(train_val_data).shuffle(FULL_SIZE)
  train_ds = train_val_ds.take(TRAIN_SIZE).map(preprocess).batch(TRAIN_SIZE)
  val_ds = train_val_ds.skip(TRAIN_SIZE).take(VAL_SIZE).map(preprocess).batch(VAL_SIZE)
  test_ds = tf.data.Dataset.from_tensor_slices(test_data).map(preprocess).batch(TEST_SIZE) 
  X_train, Y_train = next(iter(train_ds))
  X_val, Y_val = next(iter(val_ds))
  X_test, Y_test = next(iter(test_ds))
  return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def load_batches(preprocess):
  train_val_data, test_data = tf.keras.datasets.mnist.load_data()
  train_val_ds = tf.data.Dataset.from_tensor_slices(train_val_data).shuffle(FULL_SIZE)
  train_ds = train_val_ds.take(TRAIN_SIZE).map(preprocess).batch(BATCH_SIZE)
  val_ds = train_val_ds.skip(TRAIN_SIZE).take(VAL_SIZE).map(preprocess).batch(BATCH_SIZE)
  test_ds = tf.data.Dataset.from_tensor_slices(test_data).map(preprocess).batch(TEST_SIZE) 
  return train_ds, val_ds, test_ds