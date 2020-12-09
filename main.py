import argparse
import logging
import torch
import os

from settings import *
from models import simplecnn

def get_parameters():
  parser = argparse.ArgumentParser(description="This program is to solve classification task on MNIST dataset.")
  parser.add_argument('action', help='train | test | visualize')
  parser.add_argument('--model', help='model')
  parser.add_argument('--name', help='model name')
  parser.add_argument('--pretrained', help='pretrained model path')
  args = parser.parse_args()
  return args

def make_logger(name, logfile, level=logging.INFO):
  formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
  handler = logging.FileHandler(logfile)
  handler.setFormatter(formatter)

  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.addHandler(handler)

  return logger


if __name__ == '__main__':
  args = get_parameters()
  train_logger = make_logger('train', os.path.join(LOG_ROOT, 'train.log'))
  test_logger = make_logger('test', os.path.join(LOG_ROOT, 'test.log'))

  if args.action == 'train':
    model = simplecnn.SimpleCNN()
    train_logger.info('[{:s}] Start training.'.format(args.name))
    simplecnn.train_val(model, args.name, train_logger)
    train_logger.info('[{:s}] Finish training.'.format(args.name))
  
  elif args.action == 'test':
    model = torch.load(os.path.join(OUT_ROOT, args.name + '.pkl'))
    test_logger.info('[{:s}] Start testing.'.format(args.name))
    simplecnn.test(model, args.name, test_logger)
    test_logger.info('[{:s}] Finish testing.'.format(args.name))