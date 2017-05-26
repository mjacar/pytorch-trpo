import collections
import copy
import torch

import numpy as np
import scipy.signal as signal
import torch.nn as nn
import torch.optim as optim

from operator import mul
from torch import Tensor
from torch.autograd import Variable

def flatten_model_params(parameters):
  return torch.cat([param.view(1, -1) for param in parameters], 1)

def discount(x, gamma):
  """
  Compute discounted sum of future values
  out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
  """
  return signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred, y):
  """
  Var[ypred - y] / var[y].
  https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
  """
  assert y.ndim == 1 and ypred.ndim == 1
  vary = np.var(y)
  return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

class ValueFunctionWrapper(nn.Module):
  """
  Wrapper around any value function model to add fit and predict functions
  """
  def __init__(self, model):
    super(ValueFunctionWrapper, self).__init__()
    self.model = model
    self.optimizer = optim.LBFGS(self.model.parameters())

  def forward(self, data):
    return self.model.forward(data)

  def fit(self, observations, labels):
    def closure():
      predicted = self.forward(torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in observations]))
      loss = torch.pow(predicted - labels, 2).sum()
      self.optimizer.zero_grad()
      loss.backward()
      return loss
    self.optimizer.step(closure)

  def predict(self, observations):
    return self.forward(torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in observations]))

class TRPOAgent:
  def __init__(self,
               env,
               policy_model,
               value_function_model,
               gamma=0.95,
               max_timesteps=1000,
               max_episode_length=10000,
               max_kl=0.01,
               cg_damping=0.1,
               cg_iters=10,
               residual_tol=1e-10,
               use_finite_differences=True):
    """
    Instantiate a TRPO agent

    Parameters
    ----------
    env: gym.Env
      gym environment to train on
    policy_model: torch.nn.Module
      Model to use for determining the policy
      It should take a state as input and output the estimated optimal probabilities of actions
    value_function_model: torch.nn.Module
      Model to use for estimating the state values
      It should take a state as input and output the estimated value of the state
    gamma: float
      Discount factor
    max_timesteps: int
      Maximum number of timesteps in a sampled rollout
    max_episode_length: int
      Maximum number of timesteps in a single episode of a sampled rollout
    max_kl: float
      Maximum KL divergence that represents the optimization constraint
    cg_damping: float
      Coefficient for damping term in Hessian-vector product calculation
    cg_iters: int
      Number of iterations for which to run the iterative conjugate gradient algorithm
    residual_tol: float
      Residual tolerance for early stopping in the conjugate gradient algorithm
    use_finite_differences: Boolean
      Flag to indicate the use of a finite differences approximation in computing the Hessian-vector product
    """

    self.env = env
    self.policy_model = policy_model
    self.value_function_model = ValueFunctionWrapper(value_function_model)

    self.gamma = gamma
    self.max_timesteps = max_timesteps
    self.max_episode_length = max_episode_length
    self.max_kl = max_kl
    self.cg_damping = cg_damping
    self.cg_iters = cg_iters
    self.residual_tol = residual_tol
    self.use_finite_differences = use_finite_differences

    # Need to save the shape of the state_dict in order to reconstruct it from a 1D parameter vector
    self.policy_model_properties = collections.OrderedDict()
    for k, v in self.policy_model.state_dict().iteritems():
      self.policy_model_properties[k] = v.size()

  def construct_model_from_theta(self, theta):
    """
    Given a 1D parameter vector theta, return the policy model parameterized by theta
    """
    theta = theta.squeeze(0)
    new_model = copy.deepcopy(self.policy_model)
    state_dict = collections.OrderedDict()
    start_index = 0
    for k, v in self.policy_model_properties.iteritems():
      param_length = reduce(mul, v, 1)
      state_dict[k] = theta[start_index : start_index + param_length].view(v)
      start_index += param_length
    new_model.load_state_dict(state_dict)
    return new_model

  def sample_action_from_policy(self, observation):
    """
    Given an observation, return the action sampled from the policy model as well as the probabilities associated with each action
    """
    observation_tensor = Tensor(observation).unsqueeze(0)
    probabilities = self.policy_model(Variable(observation_tensor, requires_grad=True))
    action = probabilities.multinomial(1)
    return action, probabilities

  def sample_trajectories(self):
    """
    Return a rollout
    """
    paths = []
    timesteps_so_far = 0
    entropy = 0

    while timesteps_so_far < self.max_timesteps:
      observations, actions, rewards, action_distributions = [], [], [], []
      observation = self.env.reset()
      for _ in range(self.max_episode_length):
        observations.append(observation)

        action, action_dist = self.sample_action_from_policy(observation)
        actions.append(action)
        action_distributions.append(action_dist)
        entropy += -(action_dist * action_dist.log()).sum()

        observation, reward, done, info = self.env.step(action.data[0, 0])
        action.reinforce(reward)
        rewards.append(reward)

        if done:
          path = { "observations": observations,
                   "actions": actions,
                   "rewards": rewards,
                   "action_distributions": action_distributions }
          paths.append(path)
          break
      timesteps_so_far += len(path["rewards"])

    flatten = lambda l: [item for sublist in l for item in sublist]
    observations = flatten([path["observations"] for path in paths])
    discounted_rewards = flatten([discount(path["rewards"], self.gamma) for path in paths])
    actions = flatten([path["actions"] for path in paths])
    action_dists = flatten([path["action_distributions"] for path in paths])
    entropy = entropy / len(actions)

    return observations, np.asarray(discounted_rewards), actions, action_dists, entropy

  def kl_divergence(self, model):
    """
    Returns an estimate of the average KL divergence between a given model and self.policy_model
    """
    observations_tensor = torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in self.observations])
    actprob = model(observations_tensor)
    old_actprob = self.policy_model(observations_tensor)
    return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()

  def hessian_vector_product(self, vector):
    """
    Returns the product of the Hessian of the KL divergence and the given vector
    """
    self.policy_model.zero_grad()
    kl_div = self.kl_divergence(self.policy_model)
    kl_div.backward(create_graph=True)
    gradient = flatten_model_params([v.grad for v in self.policy_model.parameters()])
    gradient_vector_product = torch.sum(gradient * vector)
    gradient_vector_product.backward(torch.ones(gradient.size()))
    return (flatten_model_params([v.grad for v in self.policy_model.parameters()]) - gradient).data 

  def hessian_vector_product_fd(self, vector, r=1e-6):
    # https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
    # Estimate hessian vector product using finite differences
    # Note that this might be possible to calculate analytically in future versions of PyTorch
    vector_norm = vector.data.norm()
    theta = flatten_model_params([param for param in self.policy_model.parameters()]).data

    model_plus = self.construct_model_from_theta(theta + r * (vector.data / vector_norm))
    model_minus = self.construct_model_from_theta(theta - r * (vector.data / vector_norm))

    kl_plus = self.kl_divergence(model_plus)
    kl_minus = self.kl_divergence(model_minus)
    kl_plus.backward()
    kl_minus.backward()

    grad_plus = flatten_model_params([param.grad for param in model_plus.parameters()]).data
    grad_minus = flatten_model_params([param.grad for param in model_minus.parameters()]).data
    damping_term = self.cg_damping * vector.data
    
    return vector_norm * ((grad_plus - grad_minus) / (2 * r)) + damping_term

  def conjugate_gradient(self, b):
    """
    Returns F^(-1)b where F is the Hessian of the KL divergence
    """
    p = b.clone().data
    r = b.clone().data
    x = np.zeros_like(b.data.numpy())
    rdotr = r.dot(r)
    for i in xrange(self.cg_iters):
      if self.use_finite_differences:
        z = self.hessian_vector_product_fd(Variable(p))
      else:
        z = self.hessian_vector_product(Variable(p))
      v = rdotr / p.dot(z)
      x += v * p.numpy()
      r -= v * z
      newrdotr = r.dot(r)
      mu = newrdotr / rdotr
      p = r + mu * p
      rdotr = newrdotr
      if rdotr < self.residual_tol:
        break
    return x

  def surrogate_loss(self, theta):
    """
    Returns the surrogate loss w.r.t. the given parameter vector theta
    """
    new_model = self.construct_model_from_theta(theta.data)
    observations_tensor = torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in self.observations])
    prob_new = new_model(observations_tensor).gather(1, torch.cat(self.actions)).data
    prob_old = self.policy_model(observations_tensor).gather(1, torch.cat(self.actions)).data
    return -torch.sum((prob_new / prob_old) * self.advantage)

  def linesearch(self, x, fullstep, expected_improve_rate):
    """
    Returns the parameter vector given by a linesearch
    """
    accept_ratio = .1
    max_backtracks = 10
    fval = self.surrogate_loss(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
      xnew = x.data.numpy() + stepfrac * fullstep
      newfval = self.surrogate_loss(Variable(torch.from_numpy(xnew)))
      actual_improve = fval - newfval
      expected_improve = expected_improve_rate * stepfrac
      ratio = actual_improve / expected_improve
      if ratio > accept_ratio and actual_improve > 0:
        return Variable(torch.from_numpy(xnew))
    return x

  def step(self):
    """
    Executes an iteration of TRPO and returns the resultant policy as well as diagnostics useful for debugging
    """
    # Generate rollout
    self.observations, self.discounted_rewards, self.actions, self.action_dists, self.entropy = self.sample_trajectories()

    # Calculate the advantage of each step by taking the actual discounted rewards seen
    # and subtracting the estimated value of each state
    baseline = self.value_function_model.predict(self.observations).data
    discounted_rewards_tensor = Tensor(self.discounted_rewards).unsqueeze(1)
    advantage = discounted_rewards_tensor - baseline

    # Normalize the advantage
    self.advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    # Calculate the surrogate loss as the elementwise product of the advantage and the probability ratio of actions taken
    new_p = torch.cat(self.action_dists).gather(1, torch.cat(self.actions))
    old_p = new_p.detach()
    prob_ratio = new_p / old_p
    surrogate_objective = torch.sum(prob_ratio * Variable(self.advantage))

    # Calculate the gradient of the surrogate loss
    self.policy_model.zero_grad()
    surrogate_objective.backward(create_graph=True)
    policy_gradient = flatten_model_params([v.grad for v in self.policy_model.parameters()])

    # Use conjugate gradient algorithm to determine the step direction in theta space
    step_direction = self.conjugate_gradient(policy_gradient)
    step_direction_variable = Variable(torch.from_numpy(step_direction))

    # Do line search to determine the stepsize of theta in the direction of step_direction
    if self.use_finite_differences:
      shs = .5 * step_direction.dot(self.hessian_vector_product_fd(step_direction_variable).numpy().T)
    else:
      shs = .5 * step_direction.dot(self.hessian_vector_product(step_direction_variable).numpy().T)
    lm = np.sqrt(shs[0][0] / self.max_kl)
    fullstep = step_direction / lm
    gdotstepdir = policy_gradient.dot(step_direction_variable.t()).data[0]
    theta = self.linesearch(flatten_model_params(list(self.policy_model.parameters())), fullstep, gdotstepdir / lm)

    # Update parameters of policy model
    old_model = copy.deepcopy(self.policy_model)
    old_model.load_state_dict(self.policy_model.state_dict())
    self.policy_model = self.construct_model_from_theta(theta.data)
    kl_old_new = self.kl_divergence(old_model)

    # Fit the estimated value function to the actual observed discounted rewards
    ev_before = explained_variance_1d(baseline.squeeze(1).numpy(), self.discounted_rewards)
    self.value_function_model.fit(self.observations, Variable(Tensor(self.discounted_rewards)))
    ev_after = explained_variance_1d(self.value_function_model.predict(self.observations).data.squeeze(1).numpy(), self.discounted_rewards)

    diagnostics = collections.OrderedDict([ ('KL_Old_New', kl_old_new.data[0]), ('Entropy', self.entropy.data[0]), ('EV_Before', ev_before), ('EV_After', ev_after) ])

    return self.policy_model, diagnostics
