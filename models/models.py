import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch import Tensor
from torch.autograd import Variable

class FeedForwardSoftmax(nn.Module):
  """
  Model used for the policy model
  """
  def __init__(self, input_size, output_size):
    super(FeedForwardSoftmax, self).__init__()
    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, output_size)
    self.softmax = nn.Softmax()

    self.initialize_weights()

  def initialize_weights(self):
    init.xavier_uniform(self.fc1.weight)
    init.xavier_uniform(self.fc2.weight)

  def forward(self, data):
    output = F.tanh(self.fc1(data))
    output = self.softmax(self.fc2(output))
    return output

class ConvolutionalSoftmax(nn.Module):
  """
  Model used for the policy model
  """
  def __init__(self, output_size):
    super(ConvolutionalSoftmax, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.head = nn.Linear(448, output_size)
    self.softmax = nn.Softmax()

    self.initialize_weights()

  def initialize_weights(self):
    conv_layers = [v for k,v in self._modules.iteritems() if 'conv' in k]
    for layer in conv_layers:
      init.xavier_uniform(layer.weight)
    init.xavier_uniform(self.head.weight)

  def forward(self, data):
    output = F.selu(self.conv1(data))
    output = F.selu(self.conv2(output))
    output = F.selu(self.conv3(output))
    output = self.softmax(self.head(output.view(output.size(0), -1)))
    return output

class FeedForwardRegressor(nn.Module):
  """
  Model used for the value function model
  """
  def __init__(self, input_size):
    super(FeedForwardRegressor, self).__init__()
    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, 64)
    self.head = nn.Linear(64, 1)

    self.initialize_weights()

  def initialize_weights(self):
    init.xavier_uniform(self.fc1.weight)
    init.xavier_uniform(self.fc2.weight)
    init.xavier_uniform(self.head.weight)

  def forward(self, data):
    output = F.relu(self.fc1(data))
    output = F.relu(self.fc2(output))
    output = self.head(output)
    return output

class ConvolutionalRegressor(nn.Module):
  """
  Model used for the value function model
  """
  def __init__(self):
    super(ConvolutionalRegressor, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.head = nn.Linear(448, 1)

    self.initialize_weights()

  def initialize_weights(self):
    conv_layers = [v for k,v in self._modules.iteritems() if 'conv' in k]
    for layer in conv_layers:
      init.xavier_uniform(layer.weight)
    init.xavier_uniform(self.head.weight)

  def forward(self, data):
    output = F.selu(self.conv1(data))
    output = F.selu(self.conv2(output))
    output = F.selu(self.conv3(output))
    output = self.head(output.view(output.size(0), -1))
    return output

class DQNSoftmax(nn.Module):
  def __init__(self, output_size):
    super(DQNSoftmax, self).__init__()

    self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.fc = nn.Linear(3136, 512)
    self.head = nn.Linear(512, output_size)
    self.softmax = nn.Softmax()

    self.initialize_weights()

  def initialize_weights(self):
    conv_layers = [v for k,v in self._modules.iteritems() if 'conv' in k]
    for layer in conv_layers:
      init.xavier_uniform(layer.weight)
    init.xavier_uniform(self.head.weight)
    init.xavier_uniform(self.fc.weight)

  def forward(self, x):
    out = F.selu((self.conv1(x)))
    out = F.selu(self.conv2(out))
    out = F.selu(self.conv3(out))
    out = F.selu(self.fc(out.view(out.size(0), -1)))
    out = self.softmax(self.head(out))
    return out

class DQNRegressor(nn.Module):
  def __init__(self):
    super(DQNRegressor, self).__init__()

    self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.fc = nn.Linear(3136, 512)
    self.head = nn.Linear(512, 1)

    self.initialize_weights()

  def initialize_weights(self):
    conv_layers = [v for k,v in self._modules.iteritems() if 'conv' in k]
    for layer in conv_layers:
      init.xavier_uniform(layer.weight)
    init.xavier_uniform(self.head.weight)
    init.xavier_uniform(self.fc.weight)

  def forward(self, x):
    out = F.selu((self.conv1(x)))
    out = F.selu(self.conv2(out))
    out = F.selu(self.conv3(out))
    out = F.selu(self.fc(out.view(out.size(0), -1)))
    out = self.head(out)
    return out
