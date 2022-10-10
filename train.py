from stable_baselines3.dqn.policies import MlpPolicy
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# from environment import CustomEnv
from leaky_environment import CustomEnv

env = CustomEnv()

check_env(env)

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

logpath = "./log/"
new_logger = configure(logpath, ["stdout", "csv"])
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./save_weights/', name_prefix='rl_model')

model.learn(total_timesteps=300000, callback=checkpoint_callback)
model.save("./save_weights/learned_model")
env.close()