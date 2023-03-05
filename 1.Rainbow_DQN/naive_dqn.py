import argparse
import gym
import numpy as np
import torch

class Runner:
    def __init__(self, used_args, used_seed):
        self.args = used_args
        self.seed = used_seed
        self.env_name = self.args.env_name

        self.env = gym.make(self.env_name)
        self.eval_env = gym.make(self.env_name)   # 评估策略时需要重新make环境
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.eval_env.seed(self.seed)
        self.eval_env.action_space(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.args.obs_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_length_limit = self.env._max_episode_steps  # 对局的最大步长数
        


if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description="hyperparameter of naive dqn algorithm")
    parsers.add_argument("--max_train_steps", type=int, default=4e5, help="the maximum number of training steps")
    parsers.add_argument("--eval_freq", type=int, default=1e3, help="evaluate the policy every 'eval_freq' steps")
    parsers.add_argument("--eval_times", type=int, default=3, help="evaluate the policy 'eval_times' times each time")
    parsers.add_argument("--buffer_capacity", type=int, default=1e5, help="the maximum replay buffer capacity")
    parsers.add_argument("--batch_size", type=int, default=256)
    parsers.add_argument("--hidden_dim", type=int, default=256, help="the number of neurons in the hidden layers")
    parsers.add_argument("--lr", type=float, default=1e-4, help="the learning rate of the actor")
    parsers.add_argument("--gamma", type=float, default=0.99, help="the discount factor")
    parsers.add_argument("--epsilon_init", type=float, default=0.5, help="the initial epsilon of greedy")
    parsers.add_argument("--epsilon_min", type=float, default=0.1, help="the minimum epsilon of greedy")
    parsers.add_argument("--epsilon_decay_steps", type=int, default=1e5, help="decaying the epsilon to the minimum")
    parsers.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parsers.add_argument("--use_soft_update", type=bool, default=True, help="whether to use soft update")
    parsers.add_argument("--target_update_freq", type=int, default=200, help="hard update freq of the target network")
    parsers.add_argument("--use_lr_decay", type=bool, default=True, help="whether to use learning rate decay")
    parsers.add_argument("--grad_clip", type=float, default=10.0, help="the gradient clip range")
    parsers.add_argument("--env_name", type=str, choices=["CartPole-v1", "LunarLander-v2"], help="the environment of task")

    args = parsers.parse_args()
    for seed in [0, 10, 100]:
        runner = Runner(args, seed)



