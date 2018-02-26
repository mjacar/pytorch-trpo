import collections
import copy
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np

from utils.torch_utils import use_cuda, Tensor, Variable, ValueFunctionWrapper
import utils.math_utils as math_utils


class TRPOAgent:
  def __init__(self,
               env,
               policy_model,
               value_function_model,
               value_function_lr=1,
               gamma=0.98,
               episodes=1,
               max_kl=0.001,
               cg_damping=0.001,
               cg_iters=10,
               residual_tol=1e-10,
               ent_coeff=0.00,
               batch_size=8192):
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
    value_function_lr: float
      L-BFGS learning rate used for the value function model
    gamma: float
      Discount factor
    episodes: int
      Number of episodes to sample per iteration
    max_kl: float
      Maximum KL divergence that represents the optimization constraint
    cg_damping: float
      Coefficient for damping term in Hessian-vector product calculation
    cg_iters: int
      Number of iterations for which to run the iterative conjugate gradient algorithm
    residual_tol: float
      Residual tolerance for early stopping in the conjugate gradient algorithm
    ent_coeff: float
      Coefficient for regularizing the policy model with respect to entropy
    batch_size: int
      Batch size to train on
    """

    self.env = env
    self.policy_model = policy_model
    self.value_function_model = ValueFunctionWrapper(
        value_function_model, value_function_lr)

    if use_cuda:
      self.policy_model.cuda()
      self.value_function_model.cuda()

    self.gamma = gamma
    self.episodes = episodes
    self.max_kl = max_kl
    self.cg_damping = cg_damping
    self.cg_iters = cg_iters
    self.residual_tol = residual_tol
    self.ent_coeff = ent_coeff
    self.batch_size = batch_size

  def sample_action_from_policy(self, observation):
    """
    Given an observation, return the action sampled from the policy model as well as the probabilities associated with each action
    """
    observation_tensor = Tensor(observation).unsqueeze(0)
    probabilities = self.policy_model(
        Variable(observation_tensor, requires_grad=True))
    action = probabilities.multinomial(1)
    return action, probabilities

  def sample_trajectories(self):
    """
    Return a rollout
    """
    paths = []
    episodes_so_far = 0
    entropy = 0

    while episodes_so_far < self.episodes:
      episodes_so_far += 1
      observations, actions, rewards, action_distributions = [], [], [], []
      observation = self.env.reset()
      while True:
        observations.append(observation)

        action, action_dist = self.sample_action_from_policy(observation)
        actions.append(action)
        action_distributions.append(action_dist)
        entropy += -(action_dist * action_dist.log()).sum()

        observation, reward, done, _ = self.env.step(action.data[0, 0])
        rewards.append(reward)

        if done:
          path = {"observations": observations,
                  "actions": actions,
                  "rewards": rewards,
                  "action_distributions": action_distributions}
          paths.append(path)
          break

    def flatten(l): return [item for sublist in l for item in sublist]
    observations = flatten([path["observations"] for path in paths])
    discounted_rewards = flatten([math_utils.discount(
        path["rewards"], self.gamma) for path in paths])
    total_reward = sum(flatten([path["rewards"]
                                for path in paths])) / self.episodes
    actions = flatten([path["actions"] for path in paths])
    action_dists = flatten([path["action_distributions"] for path in paths])
    entropy = entropy / len(actions)

    return observations, np.asarray(discounted_rewards), total_reward, actions, action_dists, entropy

  def mean_kl_divergence(self, model):
    """
    Returns an estimate of the average KL divergence between a given model and self.policy_model
    """
    observations_tensor = torch.cat(
        [Variable(Tensor(observation)).unsqueeze(0) for observation in self.observations])
    actprob = model(observations_tensor).detach() + 1e-8
    old_actprob = self.policy_model(observations_tensor)
    return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()

  def hessian_vector_product(self, vector):
    """
    Returns the product of the Hessian of the KL divergence and the given vector
    """
    self.policy_model.zero_grad()
    mean_kl_div = self.mean_kl_divergence(self.policy_model)
    kl_grad = torch.autograd.grad(
        mean_kl_div, self.policy_model.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
    grad_vector_product = torch.sum(kl_grad_vector * vector)
    grad_grad = torch.autograd.grad(
        grad_vector_product, self.policy_model.parameters())
    fisher_vector_product = torch.cat(
        [grad.contiguous().view(-1) for grad in grad_grad]).data
    return fisher_vector_product + (self.cg_damping * vector.data)

  def conjugate_gradient(self, b):
    """
    Returns F^(-1)b where F is the Hessian of the KL divergence
    """
    p = b.clone().data
    r = b.clone().data
    x = np.zeros_like(b.data.cpu().numpy())
    rdotr = r.double().dot(r.double())
    for _ in xrange(self.cg_iters):
      z = self.hessian_vector_product(Variable(p)).squeeze(0)
      v = rdotr / p.double().dot(z.double())
      x += v * p.cpu().numpy()
      r -= v * z
      newrdotr = r.double().dot(r.double())
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
    new_model = copy.deepcopy(self.policy_model)
    vector_to_parameters(theta, new_model.parameters())
    observations_tensor = torch.cat(
        [Variable(Tensor(observation)).unsqueeze(0) for observation in self.observations])
    prob_new = new_model(observations_tensor).gather(
        1, torch.cat(self.actions)).data
    prob_old = self.policy_model(observations_tensor).gather(
        1, torch.cat(self.actions)).data + 1e-8
    return -torch.mean((prob_new / prob_old) * self.advantage)

  def linesearch(self, x, fullstep, expected_improve_rate):
    """
    Returns the parameter vector given by a linesearch
    """
    accept_ratio = .1
    max_backtracks = 10
    fval = self.surrogate_loss(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
      print("Search number {}...".format(_n_backtracks + 1))
      xnew = x.data.cpu().numpy() + stepfrac * fullstep
      newfval = self.surrogate_loss(Variable(torch.from_numpy(xnew)))
      actual_improve = fval - newfval
      expected_improve = expected_improve_rate * stepfrac
      ratio = actual_improve / expected_improve
      if ratio > accept_ratio and actual_improve > 0:
        return Variable(torch.from_numpy(xnew))
    return x

  def step(self):
    """
    Executes an iteration of TRPO
    """
    # Generate rollout
    all_observations, all_discounted_rewards, total_reward, all_actions, all_action_dists, self.entropy = self.sample_trajectories()

    num_batches = len(all_actions) / self.batch_size if len(
        all_actions) % self.batch_size == 0 else (len(all_actions) / self.batch_size) + 1
    for batch_num in range(num_batches):
      print("Processing batch number {}".format(batch_num+1))
      self.observations = all_observations[batch_num *
                                           self.batch_size:(batch_num+1)*self.batch_size]
      self.discounted_rewards = all_discounted_rewards[batch_num*self.batch_size:(
          batch_num+1)*self.batch_size]
      self.actions = all_actions[batch_num *
                                 self.batch_size:(batch_num+1)*self.batch_size]
      self.action_dists = all_action_dists[batch_num *
                                           self.batch_size:(batch_num+1)*self.batch_size]

      # Calculate the advantage of each step by taking the actual discounted rewards seen
      # and subtracting the estimated value of each state
      baseline = self.value_function_model.predict(self.observations).data
      discounted_rewards_tensor = Tensor(self.discounted_rewards).unsqueeze(1)
      advantage = discounted_rewards_tensor - baseline

      # Normalize the advantage
      self.advantage = (advantage - advantage.mean()) / \
          (advantage.std() + 1e-8)

      # Calculate the surrogate loss as the elementwise product of the advantage and the probability ratio of actions taken
      new_p = torch.cat(self.action_dists).gather(1, torch.cat(self.actions))
      old_p = new_p.detach() + 1e-8
      prob_ratio = new_p / old_p
      surrogate_loss = - \
          torch.mean(prob_ratio * Variable(self.advantage)) - \
          (self.ent_coeff * self.entropy)

      # Calculate the gradient of the surrogate loss
      self.policy_model.zero_grad()
      surrogate_loss.backward(retain_graph=True)
      policy_gradient = parameters_to_vector(
          [v.grad for v in self.policy_model.parameters()]).squeeze(0)

      if policy_gradient.nonzero().size()[0]:
        # Use conjugate gradient algorithm to determine the step direction in theta space
        step_direction = self.conjugate_gradient(-policy_gradient)
        step_direction_variable = Variable(torch.from_numpy(step_direction))

        # Do line search to determine the stepsize of theta in the direction of step_direction
        shs = .5 * \
            step_direction.dot(self.hessian_vector_product(
                step_direction_variable).cpu().numpy().T)
        lm = np.sqrt(shs / self.max_kl)
        fullstep = step_direction / lm
        gdotstepdir = -policy_gradient.dot(step_direction_variable).data[0]
        theta = self.linesearch(parameters_to_vector(
            self.policy_model.parameters()), fullstep, gdotstepdir / lm)

        # Fit the estimated value function to the actual observed discounted rewards
        ev_before = math_utils.explained_variance_1d(
            baseline.squeeze(1).cpu().numpy(), self.discounted_rewards)
        self.value_function_model.zero_grad()
        value_fn_params = parameters_to_vector(
            self.value_function_model.parameters())
        self.value_function_model.fit(
            self.observations, Variable(Tensor(self.discounted_rewards)))
        ev_after = math_utils.explained_variance_1d(self.value_function_model.predict(
            self.observations).data.squeeze(1).cpu().numpy(), self.discounted_rewards)
        if ev_after < ev_before or np.abs(ev_after) < 1e-4:
          vector_to_parameters(
              value_fn_params, self.value_function_model.parameters())

        # Update parameters of policy model
        old_model = copy.deepcopy(self.policy_model)
        old_model.load_state_dict(self.policy_model.state_dict())
        if any(np.isnan(theta.data.cpu().numpy())):
          print("NaN detected. Skipping update...")
        else:
          vector_to_parameters(theta, self.policy_model.parameters())

        kl_old_new = self.mean_kl_divergence(old_model)
        diagnostics = collections.OrderedDict([('Total Reward', total_reward), ('KL Old New', kl_old_new.data[0]), (
            'Entropy', self.entropy.data[0]), ('EV Before', ev_before), ('EV After', ev_after)])
        for key, value in diagnostics.iteritems():
          print("{}: {}".format(key, value))

      else:
        print("Policy gradient is 0. Skipping update...")

    return total_reward
