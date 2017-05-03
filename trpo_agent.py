import collections
import copy
import torch
import utils

import numpy as np

from models import ValueFunctionWrapper
from operator import mul
from torch import Tensor
from torch.autograd import Variable

class TRPOAgent:
  def __init__(self, env, policy_model, value_function_model):
    self.env = env
    self.policy_model = policy_model
    self.value_function_model = ValueFunctionWrapper(value_function_model)

    # Need to save the shape of the state_dict in order to reconstruct it from a 1D parameter vector
    self.policy_model_properties = collections.OrderedDict()
    for k, v in self.policy_model.state_dict().iteritems():
      self.policy_model_properties[k] = v.size()

  def get_policy(self):
    return self.policy_model

  def construct_model_from_theta(self, theta):
    # Given a 1D parameter vector theta, return the policy model parameterized by theta
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
    observation_tensor = Tensor(observation).unsqueeze(0)
    probabilities = self.policy_model(Variable(observation_tensor, requires_grad=True))
    action = probabilities.multinomial(1)
    return action, probabilities

  def sample_trajectories(self, max_timesteps, max_episode_length, gamma):
    paths = []
    timesteps_so_far = 0
    entropy = 0

    while timesteps_so_far < max_timesteps:
      observations, actions, rewards, action_distributions = [], [], [], []
      observation = self.env.reset()
      for _ in range(max_episode_length):
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
    discounted_rewards = flatten([utils.discount(path["rewards"], gamma) for path in paths])
    actions = flatten([path["actions"] for path in paths])
    action_dists = flatten([path["action_distributions"] for path in paths])
    entropy = entropy / len(actions)

    return observations, np.asarray(discounted_rewards), actions, action_dists, entropy

  def kl_divergence(self, model):
    observations_tensor = torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in self.observations])
    actprob = model(observations_tensor)
    old_actprob = self.policy_model(observations_tensor)
    return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()

  def hessian_vector_product(self, vector, r=1e-6):
    # https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
    # Estimate hessian vector product using finite differences
    # Note that this might be possible to calculate analytically in future versions of PyTorch
    vector_norm = vector.data.norm()
    theta = utils.flatten_model_params([param for param in self.policy_model.parameters()]).data

    model_plus = self.construct_model_from_theta(theta + r * (vector.data / vector_norm))
    model_minus = self.construct_model_from_theta(theta - r * (vector.data / vector_norm))

    kl_plus = self.kl_divergence(model_plus)
    kl_minus = self.kl_divergence(model_minus)
    kl_plus.backward()
    kl_minus.backward()

    grad_plus = utils.flatten_model_params([param.grad for param in model_plus.parameters()]).data
    grad_minus = utils.flatten_model_params([param.grad for param in model_minus.parameters()]).data
    damping_term = utils.cg_damping * vector.data
    
    return vector_norm * ((grad_plus - grad_minus) / (2 * r)) + damping_term

  def conjugate_gradient(self, b, cg_iters=10, residual_tol=1e-10):
    # Returns F^(-1)b where F is the Hessian of the KL divergence
    p = b.clone().data
    r = b.clone().data
    x = np.zeros_like(b.data.numpy())
    rdotr = r.dot(r)
    for i in xrange(cg_iters):
      z = self.hessian_vector_product(Variable(p))
      v = rdotr / p.dot(z)
      x += v * p.numpy()
      r -= v * z
      newrdotr = r.dot(r)
      mu = newrdotr / rdotr
      p = r + mu * p
      rdotr = newrdotr
      if rdotr < residual_tol:
        break
    return x

  def surrogate_loss(self, theta):
    new_model = self.construct_model_from_theta(theta.data)
    observations_tensor = torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in self.observations])
    prob_new = new_model(observations_tensor).gather(1, torch.cat(self.actions)).data
    prob_old = self.policy_model(observations_tensor).gather(1, torch.cat(self.actions)).data
    return -torch.sum((prob_new / prob_old) * self.advantage)

  def linesearch(self, x, fullstep, expected_improve_rate):
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

  def step(self, gamma=0.95, max_timesteps=1000, max_episode_length=10000):
    # Generate rollout
    self.observations, self.discounted_rewards, self.actions, self.action_dists, self.entropy = self.sample_trajectories(max_timesteps, max_episode_length, gamma)

    # Calculate the advantage of each step by taking the actual discounted rewards seen
    # and subtracting the estimated value of each state
    baseline = self.value_function_model.predict(self.observations).data
    discounted_rewards_tensor = Tensor(self.discounted_rewards).unsqueeze(1)
    advantage = discounted_rewards_tensor - baseline

    # Normalize the advantage
    self.advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    # Calculate the surrogate loss as the elementwise product of the advantage and the probability ratio of actions taken
    new_p = torch.cat(self.action_dists).gather(1, torch.cat(self.actions))
    old_p = torch.cat(self.action_dists).gather(1, torch.cat(self.actions))
    old_p.detach()
    prob_ratio = new_p / old_p
    surrogate_objective = torch.sum(prob_ratio * Variable(self.advantage))

    # Calculate the gradient of the surrogate loss
    self.policy_model.zero_grad()
    surrogate_objective.backward()
    policy_gradient = utils.flatten_model_params([v.grad for v in self.policy_model.parameters()])

    # Use conjugate gradient algorithm to determine the step direction in theta space
    step_direction = self.conjugate_gradient(policy_gradient)
    step_direction_variable = Variable(torch.from_numpy(step_direction))

    # Do line search to determine the stepsize of theta in the direction of step_direction
    shs = .5 * step_direction.dot(self.hessian_vector_product(step_direction_variable).numpy().T)
    lm = np.sqrt(shs[0][0] / utils.max_kl)
    fullstep = step_direction / lm
    gdotstepdir = policy_gradient.dot(step_direction_variable.t())
    theta = self.linesearch(utils.flatten_model_params(list(self.policy_model.parameters())), fullstep, gdotstepdir / lm)

    # Update parameters of policy model
    old_model = copy.deepcopy(self.policy_model)
    old_model.load_state_dict(self.policy_model.state_dict())
    self.policy_model = self.construct_model_from_theta(theta.data)
    kl_old_new = self.kl_divergence(old_model)

    # Fit the estimated value function to the actual observed discounted rewards
    ev_before = utils.explained_variance_1d(baseline.squeeze(1).numpy(), self.discounted_rewards)
    self.value_function_model.fit(self.observations, Variable(Tensor(self.discounted_rewards)))
    ev_after = utils.explained_variance_1d(self.value_function_model.predict(self.observations).data.squeeze(1).numpy(), self.discounted_rewards)

    return kl_old_new.data[0], self.entropy.data[0], ev_before, ev_after
