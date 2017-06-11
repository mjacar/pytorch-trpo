import evaluation_utils
import gym

from cartpole_wrapper import CartPoleWrapper
from models import ConvolutionalRegressor, ConvolutionalSoftmax
from trpo_agent import TRPOAgent

def main():
  env = CartPoleWrapper(gym.make('CartPole-v1'))
  policy_model = ConvolutionalSoftmax(env.action_space.n)
  value_function_model = ConvolutionalRegressor()
  agent = TRPOAgent(env, policy_model, value_function_model)

  while(True):
    policy, diagnostics = agent.step()
    r = evaluation_utils.evaluate_policy(env, policy, 10000, 1.0, 100)
    print("Evaluation avg reward = %f "% r)
    for key, value in diagnostics.iteritems():
      print("{}: {}".format(key, value))

if __name__ == "__main__":
  main()
