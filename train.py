from stable_baselines3.dqn.policies import MlpPolicy
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from environment import CustomEnv
from leaky_environment import CustomEnv_leaky

env = CustomEnv()

check_env(env)
print(env)

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

logpath = "./log/"
new_logger = configure(logpath, ["stdout", "csv"])
model.set_logger(new_logger)

stopcallback = StopTrainingOnRewardThreshold(reward_threshold=1000000, verbose=1)
eval_callback = EvalCallback(eval_env = env,
                             callback_on_new_best=stopcallback,
                             n_eval_episodes=5,
                             eval_freq=1000,
                             log_path='./save_weights/',
                             best_model_save_path='./save_weights/',
                             deterministic=True
                            )

model.learn(total_timesteps=300000, callback=eval_callback)
model.save("./save_weights/learned_model")
env.close()