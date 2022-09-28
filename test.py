from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from environment import CustomEnv

env = CustomEnv()

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="log")

# pathを指定して任意の重みをロードする
model = DQN.load("./save_weights/rl_model_10000_steps")

# 10回試行する
for i in range(10):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            break

