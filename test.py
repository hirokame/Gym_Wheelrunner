import matplotlib.pyplot as plt
import numpy as np

from sb3_contrib import RecurrentPPO

from environment import CustomEnv
from leaky_environment import CustomEnv_leaky


env = CustomEnv()
obs = env.reset()

model = RecurrentPPO.load("./save_weights/best_model")

env.popup()

# 10回試行する
for i in range(10):
    print(f"Attempt:{i+1}")
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            print(int(env.time/10))
            obs = env.reset()
            break

