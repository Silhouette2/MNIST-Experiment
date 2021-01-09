import argparse
import os
import joblib
import tensorflow as tf

from settings import *
from models import *

sklearn_models = {'svm': svm, 'knn': knn, 'logistic': logistic, 'dectree': dectree, 'bayesian': bayesian}
nn_models = {'cnn': cnn, 'resnet': resnet, 'inception': inceptionnet}

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="This program is to solve classification task on MNIST dataset.")
  parser.add_argument('action', help='train/test/visualize')
  parser.add_argument('model', help='model selection')
  parser.add_argument('path', help='path where pre-trained model is placed or will be saved')
  parser.add_argument('tag', help='model tag')
  args = parser.parse_args()

  if args.action == 'train':
    if args.model in sklearn_models:
      model_object = None
      if os.path.isfile(args.path):
        model_object = joblib.load(args.path)
      model_object = sklearn_models[args.model].train(model=model_object, tag=args.tag)
      joblib.dump(model_object, args.path)
    
    elif args.model in nn_models:
      model_object = None
      if os.path.isfile(args.path):
        model_object = tf.keras.models.load_model(args.path)
      nn_models[args.model].train(model=model_object, tag=args.tag)
      if not os.path.exists(args.path):
        os.mkdir(args.path)
      tf.keras.models.save_model(model_object, args.path)

  if args.action == 'test':
    if args.model in sklearn_models:
      model_object = joblib.load(args.path)
      sklearn_models[args.model].test(model=model_object, tag=args.tag)
    
    elif args.model in nn_models:
      model_object = tf.saved_model.load(args.path)
      nn_models[args.model].test(model=model_object, tag=args.tag)