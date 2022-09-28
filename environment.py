import gym
from gym import space
import numpy as np

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        
        self.STEP = 0 ## Turn Numberを保管
        self.maxSTEP = 200 ## 200歩走れたら成功
        
        self.ACTION = [["TORQUE UP/TORQmatplotlib アニメーション gymUE UP", "TORQUE UP/NONE", "TORQUE UP/TORQUE DOWN"]
                       ,["NONE/TORQUE UP", "NONE/NONE", "NONE/TORQUE DOWN"]
                       ,["TORQUE DOWN/TORQUE UP", "TORQUE DOWN/NONE", "TORQUE DOWN/TORQUE DOWN"]]
        
        self.action_num = len(self.ACTION)
        self.action_space = tuple((space.Discrete(self.action_num), space.Discrete(self.action_num)))
        
        # [左右の角度、角速度, ヒゲのAnticipation] [次にくるペグの本数]
        self.observation_space = tuple((space.Box(0,float("inf"),(6,)), space.Discrete(2)))
        self.reward_range = [0,float("inf")]       # 報酬の範囲[最小値と最大値]を定義
        
        self.dt = 0.01 # 10msごとの制御
        self.prepare = 0.3 # ペグが到着する何sec前にヒゲでdetectするか。
        self.frame = int(self.prepare/self.dt) #ヒゲでdetectしてから何stepでペグが到着するか。
        
        self.L_pegloc = []
        self.R_pegloc = []
        
    def reset(self):
        '''
        self.ang      : 角度 (rad)
        self.angV     : 角速度 (rad/sec)
        self.stumulus : ペグをDetect(ペグ到着の前に髭でDetect)した瞬間1→ペグが到着した瞬間0になる線形減少関数。ペグが到着するまでのカウントダウン的な役割
        self.pegloc   : ペグの場所をそれぞれのペグごとに格納したリスト。新しくDetectされたペグはStepが進むごとに-1され、到着したタイミングで0になる。
        '''
        
        self.L_ang = 0
        self.R_ang = 0
        self.L_angV = 0
        self.R_angV = 0
        
        self.L_stimulus = 0
        self.R_stimulus = 0
        
        self.L_pegloc = np.empty(0)
        self.R_pegloc = np.empty(0)
        
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
        
        self.R_ang = self.R_ang + dR_ang
        self.L_ang = self.L_ang + dL_ang
        
        ### Stimulationを更新
        self.L_pegloc.remove(-5) ## ペグが到着してから10frame過ぎたらpeglocから消去する
        self.R_pegloc.remove(-5)
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
        
        """
        int_value  : 周期の変化が少ない場合(ACTIONがNONE、つまり角加速度が0の場合)報酬をもらえる。
        phase_value: 位相の変化が少ない場合(右と左脚のAngular Velocityの差が小さい)報酬をもらえる。(angular velocityの逆数で報酬を与える)
        """
        
        if action["Left"] == 1: # NONE
            int_value += 1
        elif action["Right"] == 1: # NONE
            int_value += 1
        phase_value = 1/(0.1 + (self.R_angV - self.L_angV)**2)
        
        reward = int_value + 1/(1+phase_shift)
        
        if (self.R_ang > 2*np.pi):
            self.TURN += 1
            if len(self.R_pegloc)==0:  ## 脚をついた時にそもそもPegをDetectしてなかった時は失敗
                self.R_ang -= 2*np.pi
                reward -= 50
                done = True
            elif self.R_pegloc[0] > 5: ##　脚をついたときにPeglocationが-10〜10の間 (-50ms〜50msの間)ならばOK、それ以外なら失敗
                self.R_ang -= 2*np.pi
                reward -= 50
                done = True
            elif self.TURN >= self.maxTURN: ## max歩以上走れたときは成功、報酬を与えて終了
                reward += 100
                done = True
            else:
                self.R_ang -= 2*np.pi ## 普通に成功したときはrewardをちょっとだけ与えて続行
                reward += 1
                done = False
        else:
            done = False
        
        if (self.L_ang > 2*np.pi) and (not done):
            self.TURN += 1
            if len(self.L_pegloc)==0:  ## 脚をついた時にそもそもPegをDetectしてなかった時は失敗
                self.L_ang -= 2*np.pi
                reward -= 50
                done = True
            elif self.L_pegloc[0] > 5: ##　脚をついたときにPeglocationが-5〜5の間 (-50ms〜50msの間)ならばOK、それ以外なら失敗
                self.L_ang -= 2*np.pi
                reward -= 50
                done = True
            elif self.TURN >= self.maxTURN: ## max歩以上走れたときは成功、報酬を与えて終了
                reward += 100
                done = True
            else:
                self.L_ang -= 2*np.pi ## 普通に成功したときはrewardをちょっとだけ与えて続行
                reward += 1
                done = False
        
        obs = tuple(([self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stumulus], [self.L_pegloc, self.R_pegloc]))
        
        info = {"dR_ang":dR_ang, "dL_ang":dL_ang, "int_value":int_value, "phase_value":phase_value}
        
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        # modeとしてhuman, rgb_array, ansiが選択可能
        # humanなら描画し, rgb_arrayならそれをreturnし, ansiなら文字列をreturnする
        pass