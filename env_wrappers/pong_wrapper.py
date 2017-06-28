import gym

import numpy as np

from scipy.misc import imresize
from skimage import color

class PongWrapper(gym.Env):
  """
  Wrapper around a Atari Pong gym Env such that observations are raw pixel output
  """
  def __init__(self, env):
    super(PongWrapper, self).__init__()
    self.env = env
    self.action_space = gym.spaces.Discrete(3)
    self.action_mapping = [0, 2, 5]

  def get_screen(self):
    screen = self.env.render(mode='rgb_array')
    screen = color.rgb2gray(screen)
    screen = imresize(screen, (110, 84))
    screen = screen[18:102][:] / 255.0
    return screen.astype(np.float)

  def reset(self):
    self.env.reset()
    for _ in range(20):
      self.env.step(0)
    screen = self.get_screen()
    return np.asarray([screen, screen, screen, screen]).astype(np.float)

  def step(self, action):
    screens = []
    total_reward = 0

    for t in range(4):
      screen = self.get_screen()
      screens.append(screen)
      _, reward, done, info = self.env.step(self.action_mapping[action])
      total_reward += reward
      if done or total_reward:
        if done:
          self.env.reset()
        for _ in range(20):
          self.env.step(0)
        for _ in range(3 - t):
          screens.append(screen)
        break

    screens = np.asarray(screens).astype(np.float)
    return screens, total_reward, done, info
