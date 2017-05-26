import torch

from torch.autograd import Variable

def evaluate_policy(env, policy, maximum_episode_length=100000, discount_factor=0.95, nb_episodes=1):
  """
  Evaluate a policy over multiple trajectories
  Params:
    - env: the environement
    - policy: the policy to evaluate
    - maximum_episode_length: the maximum length of the trajectory
    - discount_factor: the discount_factor
    - nb_episodes: the number of episodes to sample
  """
  r = 0
  for _ in range(nb_episodes):
    r += evaluate_episode(env, policy, maximum_episode_length=maximum_episode_length, discount_factor=discount_factor)
  return r / nb_episodes

def evaluate_episode(env, policy, maximum_episode_length=100000, discount_factor=0.95):
  reward = 0
  df = 1.0
  observation = env.reset()
  for _ in range(maximum_episode_length):
    action = policy(Variable(torch.from_numpy(observation).float().unsqueeze(0))).max(1)[1].data[0, 0]
    observation, immediate_reward, finished, info = env.step(action)
    reward = reward + df * immediate_reward
    df = df * discount_factor
    if finished:
      break
  return reward
