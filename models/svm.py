from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.process import sklearn_test, sklearn_train
from utils.features import *

def feature(x, y):
  return collect_features(x), y

def train(model=None, tag='default', **model_args):
  if model == None:
    model = make_pipeline(StandardScaler(), SVC(**model_args))
  sklearn_train(model, feature, tag)
  return model

def test(model, tag='default'):
  sklearn_test(model, feature, tag)