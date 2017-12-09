import argparse
import subprocess
import torch

from itertools import count
from models import DQNRegressor, DQNSoftmax
from tensorboard_logger import configure, log_value
from trpo_agent import TRPOAgent
from utils.atari_wrapper import make_atari, wrap_deepmind

def main(env_id):
  env = wrap_deepmind(make_atari(env_id), scale=True)
  policy_model = DQNSoftmax(env.action_space.n)
  value_function_model = DQNRegressor()
  agent = TRPOAgent(env, policy_model, value_function_model)

  subprocess.Popen(["tensorboard", "--logdir", "runs"])
  configure("runs/pong-run")

  for t in count():
    reward = agent.step()
    log_value('score', reward, t)
    if t % 100 == 0:
      torch.save(policy_model.state_dict(), "policy_model.pth")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
  args = parser.parse_args()
  main(args.env)
