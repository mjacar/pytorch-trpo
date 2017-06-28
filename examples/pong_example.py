import gym

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from utils.evaluation_utils import evaluate_policy
from env_wrappers.pong_wrapper import PongWrapper
from models.models import DQNRegressor, DQNSoftmax
from trpo_agent import TRPOAgent

def main():
  env = PongWrapper(gym.make('Pong-v0'))
  policy_model = DQNSoftmax(env.action_space.n)
  value_function_model = DQNRegressor()
  agent = TRPOAgent(env, policy_model, value_function_model, value_function_lr=0.25)

  while(True):
    policy, diagnostics = agent.step()
    r = evaluate_policy(env, policy, discount_factor=1.0)
    print("Evaluation avg reward = %f "% r)
    for key, value in diagnostics.iteritems():
      print("{}: {}".format(key, value))

if __name__ == "__main__":
  main()
