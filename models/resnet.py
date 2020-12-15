import tensorflow as tf
from utils.loggers import test_logger, train_logger

class ResBlock(tf.keras.Model):
  def __init__(self, fn, k):
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=fn, kernel_size=k, padding='same', activation='relu')
    self.conv2 = tf.keras.layers.Conv2D(filters=fn, kernel_size=k, padding='same')
    self.relu = tf.keras.layers.ReLU()
  
  def call(self, x):
    r = self.conv1(x)
    r = self.conv2(r)
    return self.relu(x + r)

class ResNet(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')
    self.res11 = ResBlock(fn=16, k=3)
    self.res12 = ResBlock(fn=16, k=3)
    self.res13 = ResBlock(fn=16, k=3)
    self.pool = tf.keras.layers.MaxPool2D()
    self.res21 = ResBlock(fn=16, k=3)
    self.res22 = ResBlock(fn=16, k=3)
    self.res23 = ResBlock(fn=16, k=3)
    self.pool = tf.keras.layers.MaxPool2D()
    self.res31 = ResBlock(fn=16, k=3)
    self.res32 = ResBlock(fn=16, k=3)
    self.res33 = ResBlock(fn=16, k=3)
    self.flatten = tf.keras.layers.Flatten()
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.fc = tf.keras.layers.Dense(10, activation='softmax')
  
  def call(self, x):
    x = self.conv(x)
    x = self.res11(x)
    x = self.res12(x)
    x = self.res13(x)
    x = self.pool(x)
    x = self.res21(x)
    x = self.res22(x)
    x = self.res23(x)
    x = self.pool(x)
    x = self.res31(x)
    x = self.res32(x)
    x = self.res33(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.fc(x)
    return x



def __transform(x, y):
  x = tf.cast(x, tf.float32)
  x = x / 255.0
  return tf.expand_dims(x,-1), y


def train(model=None, tag='default'):

  with tf.device(tf.test.gpu_device_name()):

    # Load the dataset.
    train_val_data, _ = tf.keras.datasets.mnist.load_data()
    train_val_ds = tf.data.Dataset.from_tensor_slices(train_val_data).shuffle(60000)
    train_ds = train_val_ds.take(10000).map(__transform).batch(8)
    val_ds = train_val_ds.skip(10000).take(5000).map(__transform).batch(8)

    # If model is not specified, create a new one.
    if model == None:
      model = ResNet()
    model._set_inputs(tf.random.normal([8,28,28,1]))
    
    # The loss object.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Compute the current loss.
    def loss(model, x, y):
      p = model(x)
      return loss_object(y, p)

    # The optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Compute the gradients.
    def grad(model, x, y):
      with tf.GradientTape() as tape:
        loss_value = loss(model, x, y)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # Start training.
    loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in range(8):
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

    return model


def test(model, tag='default'):

  with tf.device(tf.test.gpu_device_name()):
  
    _, test_data = tf.keras.datasets.mnist.load_data()
    test_ds = tf.data.Dataset.from_tensor_slices(test_data).map(__transform).batch(8)

    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for batch, (x, y) in enumerate(test_ds):
      accuracy(y, model(x))

    test_logger.critical('[{:s}] Test accuracy: {:.3%}'.format(tag, accuracy.result()))