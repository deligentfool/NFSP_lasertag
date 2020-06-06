from nfsp import nfsp
import gym
import lasertag
from wrapper import env_wrap
import torch


if __name__ == '__main__':
    env = gym.make('LaserTag-small2-v0')
    env = env_wrap(env)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = nfsp(
        env=env,
        epsilon_init=1.,
        decay=30000,
        epsilon_min=0.01,
        update_freq=1000,
        sl_lr=1e-4,
        rl_lr=1e-4,
        sl_capa=100000,
        rl_capa=50000,
        n_step=1,
        gamma=0.95,
        eta=0.1,
        max_episode=1000,
        negative=False,
        rl_start=10000,
        sl_start=1000,
        train_freq=1,
        rl_batch_size=128,
        sl_batch_size=128,
        render=False,
        device=device
    )
    test.run()