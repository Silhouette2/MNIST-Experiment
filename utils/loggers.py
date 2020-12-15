import os
import logging
from settings import *

def make_logger(name, logfile, level=logging.INFO):
  formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
  handler = logging.FileHandler(logfile)
  handler.setFormatter(formatter)

  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.addHandler(handler)

  return logger


train_logger = make_logger('train', os.path.join(LOG_ROOT, 'train.log'))
test_logger = make_logger('test', os.path.join(LOG_ROOT, 'test.log'))
