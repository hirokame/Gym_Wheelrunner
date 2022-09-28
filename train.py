from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.callbacks import CheckpointCallback

from environment import CustomEnv

env = CustomEnv()

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="log")
print('start learning')
checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./save_weights/', name_prefix='rl_model')
model.learn(total_timesteps=10000, callback=checkpoint_callback)
print('finish learning')