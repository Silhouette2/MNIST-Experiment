from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils.process import *
from utils.features import *

def feature(x, y):
  return collect_features(x), y

def train(model=None, tag='default'):
  if model == None:
    model = make_pipeline(StandardScaler(), LogisticRegression())
  sklearn_train(model, feature, tag)
  return model

def test(model, tag='default'):
  sklearn_test(model, feature, tag)