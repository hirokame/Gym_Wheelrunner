import gym
from gym import spaces
import cv2
import numpy as np

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.ACTION = [["TORQUE UP/TORQUE UP", "TORQUE UP/NONE", "TORQUE UP/TORQUE DOWN"]
                       ,["NONE/TORQUE UP", "NONE/NONE", "NONE/TORQUE DOWN"]
                       ,["TORQUE DOWN/TORQUE UP", "TORQUE DOWN/NONE", "TORQUE DOWN/TORQUE DOWN"]]
        
        self.action_num = len(self.ACTION)
        self.action_space = tuple((gym.space.Discrete(self.action_num), gym.space.Discrete(self.action_num)))
        
        # [左右の角度、角速度, ヒゲのAnticipation] [次にくるペグの本数]
        self.observation_space = tuple((gym.space.Box(0,float("inf"),(6,)), gym.space.Discrete(2)))
        self.reward_range = [0,float("inf")]       # 報酬の範囲[最小値と最大値]を定義
        
        self.dt = 0.01 # 10msごとの制御
        self.prepare = 0.3 # ペグが到着する何sec前にヒゲでdetectするか。
        
        self.L_upcoming = []
        self.R_upcoming = []
        
    def reset(self):
        '''
        self.ang: 角度 (rad)
        self.angV: 角速度 (rad/sec)
        self.stumulus: ペグをDetect(ペグ到着の前に髭でDetect)した瞬間1→ペグが到着した瞬間0になる線形減少関数。ペグが到着するまでのカウントダウン的な役割
        '''
        self.L_ang = 0
        self.R_ang = 0
        self.L_angV = 0
        self.R_angV = 0
        
        self.L_stimulus = 0
        self.R_stimulus = 0
        
        self.L_detected = False
        self.R_detected = False
        
        obs = [self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stimulus, self.L_detected, self.R_detected]
        
        return obs

    def step(self, action):
        ### 左右の脚の角速度を更新
        if action["Left"] == 0: # TORQUE UP
            self.L_angV += 0.1
        elif action["Left"] == 2 # TORQUE DOWN
            self.L_angV -= 0.1
            
        elif action["Right"] == 0: # TORQUE UP
            self.R_angV += 0.1
        elif action["Right"] == 2: # TORQUE DOWN
            self.R_angV -= 0.1
        http://www.numpy.pi()/
        ### 左右の脚の角度(位相)を更新
        dR_ang = self.dt * self.R_angV
        dL_ang = self.dt * self.L_angV
        
        self.R_ang = (self.R_ang + dR_ang)%(np.pi*2)
        self.L_ang = (self.L_ang + dL_ang)%(np.pi*2)
        
        ### Anticipationを更新
        self.L_upcoming.remove(0)
        self.R_upcoming.remove(0)
        if len(self.L_upcoming) != 0:
            for i in range(len(self.L_upcoming)):
                self.L_upcoming[i] -= 1
        if len(self.R_upcoming) != 0:
            for i in range(len(self.R_upcoming)):
                self.R_upcoming[i] -= 1
                
        self.L_stimulus -= self.gain * len(self.L_upcoming)
        self.R_stimulus -= self.gain * len(self.R_upcoming)
        
        if self.L_detected:
            self.L_upcoming.append(10)
            self.L_stimulus += 1
        elif self.R_detected:
            self.R_upcoming.append(10)
            self.R_stimulus += 1
        
        obs = obs = [self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stumulus]
        
        """
        intshift: 
        """
        if action["Left"] == 1: # NONE
            intshift += 1
        elif action["Right"] == 1: # NONE
            intshift += 1
        
        phaseshift = (self.R_angV - self.L_angV)**2
        
        return obs, reward, done, info

    def render(self, mode='human'):   
        # modeとしてhuman, rgb_array, ansiが選択可能
        # humanなら描画し, rgb_arrayならそれをreturnし, ansiなら文字列をreturnする
        ...
  
    def close(self):
        ...

    def seed(self, seed=None):
        ...