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
        self.frame = int(self.prepare/self.dt) #ヒゲでdetectしてから何stepでペグが到着するか。
        
        self.L_pegloc = []
        self.R_pegloc = []
        
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
        
        self.L_pegloc = False
        self.R_pegloc = False
        
        obs = [self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stimulus, self.L_pegloc, self.R_pegloc]
        
        return obs

    def step(self, action):
        ### 左右の脚の角速度を更新
        if action["Left"] == 0: # TORQUE UP
            self.L_angV += 1
        elif action["Left"] == 2 # TORQUE DOWN
            self.L_angV -= 1
            
        elif action["Right"] == 0: # TORQUE UP
            self.R_angV += 1
        elif action["Right"] == 2: # TORQUE DOWN
            self.R_angV -= 1

        ### 左右の脚の角度(位相)を更新
        dR_ang = self.dt * self.R_angV
        dL_ang = self.dt * self.L_angV
        
        self.R_ang = (self.R_ang + dR_ang)%(np.pi*2)
        self.L_ang = (self.L_ang + dL_ang)%(np.pi*2)
        
        ### Stimulationを更新
        self.L_pegloc.remove(-10) ## ペグが到着してから10frame過ぎたらpeglocから消去する
        self.R_pegloc.remove(-10)
        if len(self.L_pegloc) != 0:
            for i in range(len(self.L_pegloc)):
                self.L_pegloc[i] -= 1
        if len(self.R_pegloc) != 0:
            for i in range(len(self.R_pegloc)):
                self.R_pegloc[i] -= 1
        
        L_upcoming = len(self.L_pegloc[np.where(self.pegloc)])
        self.L_stimulus -= self.gain * L_upcoming
        self.R_stimulus -= self.gain * R_upcoming
        
        if self.L_detected:
            self.L_pegloc.append(self.frame)
            self.L_stimulus += 1
        elif self.R_detected:
            self.R_pegloc.append(self.frame)
            self.R_stimulus += 1
        
        obs = obs = [self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stumulus]
        
        """
        int_value  : 周期の変化が少ない場合(ACTIONがNONE、つまり角加速度が0の場合)報酬をもらえる。
        phase_value: 位相の変化が少ない場合(右と左脚のAngular Velocityの差が小さい)報酬をもらえる。(angular velocityの逆数で報酬を与える)
        failed     : ペグが来ている場合に
        """
        if action["Left"] == 1: # NONE
            int_value += 0.5
        elif action["Right"] == 1: # NONE
            int_value += 0.5
        phase_value = 1/(1 + (self.R_angV - self.L_angV)**2)
        
        reward = int_value + 1/(1+phase_shift)
        
        if failed:
            reward -= 50
            done = True
        
        return obs, reward, done, info

    def render(self, mode='human'):   
        # modeとしてhuman, rgb_array, ansiが選択可能
        # humanなら描画し, rgb_arrayならそれをreturnし, ansiなら文字列をreturnする
        ...
  
    def close(self):
        ...

    def seed(self, seed=None):
        ...