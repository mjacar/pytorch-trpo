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
  def __init__(self, input_size, output_layer_size):
    super(FeedForwardSoftmax, self).__init__()
    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, output_layer_size)
    self.softmax = nn.Softmax()

    self.initialize_weights()

  def initialize_weights(self):
    init.xavier_uniform(self.fc1.weight)
    init.xavier_uniform(self.fc2.weight)

  def forward(self, data):
    output = F.tanh(self.fc1(data))
    output = self.softmax(self.fc2(output))
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

class ValueFunctionWrapper(nn.Module):
  """
  Wrapper around any value function model to add fit and predict functions
  """
  def __init__(self, model):
    super(ValueFunctionWrapper, self).__init__()
    self.model = model
    self.optimizer = optim.Adam(self.model.parameters())

  def forward(self, data):
    return self.model.forward(data)

  def fit(self, observations, labels):
    predicted = self.forward(torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in observations]))
    loss = torch.pow(predicted - labels, 2).sum()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def predict(self, observations):
    return self.forward(torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in observations]))
