import tensorflow as tf

def get_projection(img):
  def pj(x, axis):
    return tf.reduce_sum(tf.cast(tf.reduce_sum(x, axis) > 0, tf.uint8))
  
  return [pj(img[:7,:14], 0),    pj(img[:7,14:], 0),
          pj(img[7:14,:14], 0),  pj(img[7:14,14:], 0),
          pj(img[14:21,:14], 0), pj(img[14:21,14:], 0),
          pj(img[21:,:14], 0),   pj(img[21:,14:], 0),
          pj(img[:14,:7], 1),    pj(img[14:,:7], 1),
          pj(img[:14,7:14], 1),  pj(img[14:,7:14], 1),
          pj(img[:14,14:21], 1), pj(img[14:,14:21], 1),
          pj(img[:14,21:], 1),   pj(img[14:,21:], 1)]

def get_distribution(img):
  return [tf.reduce_mean(tf.cast(img[0:9, 0:9] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[0:9, 9:19] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[0:9, 19:28] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[9:19, 0:9] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[9:19, 9:19] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[9:19, 19:28] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[19:28, 0:9] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[19:28, 9:19] > 0, tf.uint8)),
          tf.reduce_sum(tf.cast(img[19:28, 19:28] > 0, tf.uint8))]

def collect_features(img):
  f_proj = get_projection(img)
  f_dist = get_distribution(img)
  return tf.convert_to_tensor(f_proj + f_dist)