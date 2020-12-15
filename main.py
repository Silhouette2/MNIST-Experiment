import argparse
import os
import joblib
import tensorflow as tf

from settings import *
from models import cnn, svm, resnet

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="This program is to solve classification task on MNIST dataset.")
  parser.add_argument('action', default=None, help='train/test/visualize')
  parser.add_argument('model', default=None, help='model selection')
  parser.add_argument('path', default=None, help='path where pre-trained model is placed or will be saved')
  parser.add_argument('--tag', default='default', help='model tag')
  parser.add_argument('--kernel', default='linear', help='kernel for svm: linear/poly/rbf')
  args = parser.parse_args()

  if args.action == 'train':
    if args.model == 'svm':
      model = None
      if os.path.isfile(args.path):
        model = joblib.load(args.path)
      model = svm.train(model=model, tag=args.tag, kernel=args.kernel)
      joblib.dump(model, args.path)
    elif args.model in ['cnn', 'resnet']:
      model = None
      if os.path.isfile(args.path):
        model = tf.keras.models.load_model(args.path)
      if args.model == 'cnn':
        model = cnn.train(model=model, tag=args.tag)
      elif args.model == 'resnet':
        model = resnet.train(model=model, tag=args.tag)
      if not os.path.exists(args.path):
        os.mkdir(args.path)
      tf.keras.models.save_model(model, args.path)
  if args.action == 'test':
    if args.model == 'svm': 
      model = joblib.load(args.path)
      svm.test(model=model, tag=args.tag)
    elif args.model in ['cnn', 'resnet']:
      model = tf.saved_model.load(args.path)
      cnn.test(model=model, tag=args.tag)
