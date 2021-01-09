import numpy as np
from settings import *
from utils.loggers import *
from utils.dataloader import *

def sklearn_train(model, preprocess, tag):
  ''' training sklearn models '''
  (X_train, Y_train), (X_val, Y_val), _ = load_full(preprocess)
  model.fit(X_train, Y_train)
  P_val = model.predict(X_val)
  train_logger.critical('[{:s}] Validation Accuracy: {:.3%}'.format(tag, (P_val == Y_val.numpy()).sum() / P_val.shape[0]))

def sklearn_test(model, preprocess, tag):
  ''' prediction for sklearn models '''
  print(model)
  _, _, (X_test,Y_test) = load_full(preprocess)
  P_test = model.predict(X_test)
  test_logger.critical('[{:s}] Test Accuracy: {:.3%}'.format(tag, (P_test == Y_test.numpy()).sum() / P_test.shape[0]))



def nn_train(model, preprocess, tag, lr=0.01):
  ''' training neural networks '''
  with tf.device(tf.test.gpu_device_name()):
    model._set_inputs(tf.random.normal([BATCH_SIZE, 28, 28, 1]))

    # Load the dataset.
    train_ds, val_ds, _ = load_batches(preprocess)

    # The loss object.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compute the current loss.
    def loss(model, x, y):
      p = model(x)
      return loss_object(y, p)

    # The optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    # Compute the gradients.
    def grad(model, x, y):
      with tf.GradientTape() as tape:
        loss_value = loss(model, x, y)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # Start training.
    loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in range(EPOCH_NUM):
      loss_avg.reset_states()
      accuracy.reset_states()
      for batch, (x,y) in enumerate(train_ds):
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_avg(loss_value)
        accuracy(y, model(x))

      train_logger.info("[{:s}] Epoch {:03d} | Training loss {:.5f}  Accuracy {:.3%}"
                        .format(tag, epoch, loss_avg.result(), accuracy.result()))
    
    # Validate.
    accuracy.reset_states()
    for batch, (x,y) in enumerate(val_ds):
      accuracy(y, model(x))
    train_logger.critical('[{:s}] Validation accuracy: {:.3%}'.format(tag, accuracy.result()))


def nn_test(model, preprocess, tag):
  ''' testing neural networks '''
  with tf.device(tf.test.gpu_device_name()):
    _, _, test_ds = load_batches(preprocess)
    
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for batch, (x, y) in enumerate(test_ds):
      accuracy(y, model(x))

    test_logger.critical('[{:s}] Test accuracy: {:.3%}'.format(tag, accuracy.result()))