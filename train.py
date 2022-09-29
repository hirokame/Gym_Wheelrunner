from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from environment import CustomEnv

from stable_baselines3.common.env_checker import check_env

env = CustomEnv()

check_env(env)

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="log")
print('start learning')
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./save_weights/', name_prefix='rl_model')
model.learn(total_timesteps=100000, callback=checkpoint_callback)
print('finish learning')