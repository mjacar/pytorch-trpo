import evaluation
import gym
import utils

from models import FeedForwardRegressor, FeedForwardSoftmax
from trpo_agent import TRPOAgent

def main():
  env = gym.make('CartPole-v1')
  policy_model = FeedForwardSoftmax(env.observation_space.shape[0], env.action_space.n)
  value_function_model = FeedForwardRegressor(env.observation_space.shape[0])
  agent = TRPOAgent(env, policy_model, value_function_model)

  while(True):
    kl_old_new, entropy, ev_before, ev_after = agent.step()
    policy = agent.get_policy()
    r = evaluation.evaluate_policy(env, policy, 10000, 1.0, 100)
    print("Evaluation avg reward = %f "% r)
    print("KL_Old_New: {}".format(kl_old_new))
    print("Entropy: {}".format(entropy))
    print("EV_Before: {}".format(ev_before))
    print("EV_After: {}\n".format(ev_after))

if __name__ == "__main__":
  main()
