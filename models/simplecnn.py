import os

import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as tf
from settings import *
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import MNIST


class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.feature = nn.Sequential(
      nn.Conv2d(1, 32, 3, 1, 1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3, 1, 1),
      nn.ReLU(True),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 32, 3, 1, 1),
      nn.ReLU(True),
      nn.Flatten()
    )

    self.classify = nn.Sequential(
      nn.Linear(1568, 10),
      nn.Softmax(1)
    )
    
  def forward(self, x):
    x = self.feature(x)
    x = self.classify(x)
    return x

preprocess = tf.Compose([tf.ToTensor(), tf.ConvertImageDtype(torch.float)])

def train_val(model, model_name, logger, lr=0.01, momentum=0.8, random_state=42):
  train_val_set = MNIST(root=DATA_ROOT, train=True, transform=preprocess)
  train_set, val_set = random_split(train_val_set, [TRAIN_SIZE, VAL_SIZE], generator=torch.Generator().manual_seed(random_state))
  train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
  val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)


  criterion = nn.CrossEntropyLoss()
  optimizer = opt.SGD(model.parameters(), lr=lr, momentum=momentum)

  running_loss = 0
  group_size = TRAIN_SIZE / BATCH_SIZE / 10
  for epoch in range(EPOCH_NUM):
    for batch, data in enumerate(train_loader):
      X, y = data
      optimizer.zero_grad()
      output = model(X)

      loss = criterion(output, y)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if batch % group_size == 0:
        logger.info('[{:s}] Epoch {:d} Batch {:d} Running Loss: {:f}'.format(model_name, epoch, batch, running_loss/group_size))
        running_loss = 0

  error = total = 0
  with torch.no_grad():
    for data in val_loader:
      X, y = data
      out = model(X)
      _, p = torch.max(out, 1)
      error += (p!=y).sum()
      total += p.size()[0]
    logger.info('[{:s}] Validation error rate: {:f}'.format(model_name, error / total))

  torch.save(model, os.path.join(OUT_ROOT, model_name + '.pkl'))

def test(model, model_name, logger):
  test_set = MNIST(root=DATA_ROOT, train=False, transform=preprocess)
  test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
  
  error = total = 0
  with torch.no_grad():
    for data in test_loader:
      X, y = data
      out = model(X)
      _, p = torch.max(out, 1)
      error += (p!=y).sum()
      total += p.size()[0]
    logger.info('[{:s}] Test error rate: {:f}'.format(model_name, error / total))