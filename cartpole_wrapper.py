import gym
import torch

import numpy as np
import torchvision.transforms as T

from PIL import Image

class CartPoleWrapper(gym.Env):
  """
  Wrapper around a CartPole gym Env such that observations are raw pixel output
  """
  def __init__(self, env):
    super(CartPoleWrapper, self).__init__()
    self.env = env.unwrapped
    self.resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    self.screen_width = 600
    self.action_space = self.env.action_space

  def get_cart_location(self):
    world_width = self.env.x_threshold * 2
    scale = self.screen_width / world_width
    return int(self.env.state[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART

  def get_screen(self):
    screen = self.env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = self.get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (self.screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return self.resize(screen).numpy()

  def reset(self):
    self.env.reset()
    return self.get_screen()

  def step(self, action):
   last_screen = self.get_screen()
   _, reward, done, info = self.env.step(action)
   return self.get_screen() - last_screen, reward, done, info
