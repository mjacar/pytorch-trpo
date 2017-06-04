import evaluation_utils
import gym

from models import FeedForwardRegressor, FeedForwardSoftmax
from trpo_agent import TRPOAgent

def main():
  env = gym.make('CartPole-v1')
  policy_model = FeedForwardSoftmax(env.observation_space.shape[0], env.action_space.n)
  value_function_model = FeedForwardRegressor(env.observation_space.shape[0])
  agent = TRPOAgent(env, policy_model, value_function_model)

  while(True):
    policy, diagnostics = agent.step()
    r = evaluation_utils.evaluate_policy(env, policy, 10000, 1.0, 100)
    print("Evaluation avg reward = %f "% r)
    for key, value in diagnostics.items():
      print("{}: {}".format(key, value))

if __name__ == "__main__":
  main()
