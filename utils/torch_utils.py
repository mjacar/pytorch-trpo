import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

use_cuda = torch.cuda.is_available()


def Variable(tensor, *args, **kwargs):
  if use_cuda:
    return torch.autograd.Variable(tensor, *args, **kwargs).cuda()
  else:
    return torch.autograd.Variable(tensor, *args, **kwargs)


def Tensor(nparray):
  if use_cuda:
    return torch.Tensor(nparray).cuda()
  else:
    return torch.Tensor(nparray)


class ValueFunctionWrapper(nn.Module):
  """
  Wrapper around any value function model to add fit and predict functions
  """

  def __init__(self, model, lr):
    super(ValueFunctionWrapper, self).__init__()
    self.model = model
    self.loss_fn = nn.MSELoss()
    self.lr = lr

  def forward(self, data):
    return self.model.forward(data)

  def fit(self, observations, labels):
    def closure():
      predicted = self.predict(observations)
      loss = self.loss_fn(predicted, labels)
      self.optimizer.zero_grad()
      loss.backward()
      return loss
    old_params = parameters_to_vector(self.model.parameters())
    for lr in self.lr * .5**np.arange(10):
      self.optimizer = optim.LBFGS(self.model.parameters(), lr=lr)
      self.optimizer.step(closure)
      current_params = parameters_to_vector(self.model.parameters())
      if any(np.isnan(current_params.data.cpu().numpy())):
        print("LBFGS optimization diverged. Rolling back update...")
        vector_to_parameters(old_params, self.model.parameters())
      else:
        return

  def predict(self, observations):
    return self.forward(torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in observations]))
