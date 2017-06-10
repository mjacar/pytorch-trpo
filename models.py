import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch import Tensor
from torch.autograd import Variable

class FeedForwardSoftmax(nn.Module):
  """
  Model used for the multinomial policy
  """
  def __init__(self, num_inputs, num_outputs):
    super(FeedForwardSoftmax, self).__init__()
    self.fc1 = nn.Linear(num_inputs, 64)
    self.fc2 = nn.Linear(64, num_outputs)
    self.softmax = nn.Softmax()

    self.initialize_weights()

  def initialize_weights(self):
    init.xavier_uniform(self.fc1.weight)
    init.xavier_uniform(self.fc2.weight)

  def forward(self, data):
    output = F.tanh(self.fc1(data))
    output = self.softmax(self.fc2(output))
    return output

class FeedForwardGaussian(nn.Module):
  """
  Model used for the Gaussian policy
  """
  def __init__(self, num_inputs, num_outputs):
    super(FeedForwardGaussian, self).__init__()
    self.fc1 = nn.Linear(num_inputs, 64)
    self.fc2 = nn.Linear(64, 64)
    self.action_mean = nn.Linear(64, num_outputs)

    self.action_mean.weight.data.mul_(0.1)
    self.action_mean.bias.data.mul_(0.0)

    self.action_std = nn.Parameter(torch.zeros(1, num_outputs))

  def forward(self, data):
    x = F.tanh(self.fc1(data))
    x = F.tanh(self.fc2(x))
    action_mean = self.action_mean(x)
    action_std = self.action_std.expand_as(action_mean)
    return action_mean, action_std + 1e-9

class FeedForwardRegressor(nn.Module):
  """
  Model used for the value function model
  """
  def __init__(self, num_inputs):
    super(FeedForwardRegressor, self).__init__()
    self.fc1 = nn.Linear(num_inputs, 64)
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
