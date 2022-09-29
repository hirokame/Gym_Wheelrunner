import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3 import DQN
from environment import CustomEnv


env = CustomEnv()
obs = env.reset()

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="log")
model = DQN.load("./save_weights/rl_model_70000_steps")

env.popup()

# 10回試行する
for i in range(10):
    print(f"Attempt:{i+1}")
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            break

