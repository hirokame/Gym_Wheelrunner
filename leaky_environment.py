import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.tuple import Tuple

import cv2
import time
from copy import copy



class CustomEnv_leaky(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, pegpattern="Complex"):
        super(CustomEnv_leaky, self).__init__()
        
        self.time = 0 # Timeを保管
        self.STEP = 0 # Step Numberを保管
        self.maxSTEP = 200 # 200歩走れたら成功
        
        self.ACTION = ["TORQUE SMALL UP","NONE","TORQUE LARGE UP"]
        self.action = dict({"Left":1, "Right":1})
        self.action_num = len(self.ACTION)
        self.action_space = Discrete(self.action_num**2)
        
        # [左右の角度、角速度, ヒゲのAnticipation, 次にくるペグの本数]
        low = np.array([0,0,0,0,-1,-1,0,0,0])
        high = np.array([7,7,float("inf"),float("inf"),5,5,1,1,4000])
        
        self.observation_space = Box(low, high, (9,), dtype="float32")
        self.reward_range = [0,1000]       # 報酬の範囲[最小値と最大値]を定義
        
        self.dt = 0.02 # 20msごとの制御
        self.prepare = 0.5 # ペグが到着する何sec前にヒゲでdetectするか。
        self.frame = int(self.prepare/self.dt) #ヒゲでdetectしてから何stepでペグが到着するか。
        
        self.L_pegloc = []
        self.R_pegloc = []
        
        self.set_pegpattern(pattern="Complex")
        self.print_failed_reason = False
        
    def popup(self):
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.img = np.full([1200, 1200, 3], 255, dtype="int16")
        self.im = self.ax.imshow(self.img)
        plt.show(block=False)
        
    def set_pegpattern(self, pattern="Complex"):
        if pattern=="Complex":
            self.oneturn = 4000
            self.Lpeg = [0,150,400,600,900,1300,1650,2050,2350,2750,3200,3600]
            self.Rpeg = [100,250,500,850,1150,1450,1850,2250,2400,2800,3250,3650]
            self.Ldet = sorted(list(map(lambda x:(x+4000-int(self.prepare*1000))%4000, self.Lpeg)))
            self.Rdet = sorted(list(map(lambda x:(x+4000-int(self.prepare*1000))%4000, self.Rpeg)))
        
    def reset(self):
        '''
        self.ang      : 角度 (rad)
        self.angV     : 角速度 (rad/sec)
        self.stumulus : ペグをDetect(ペグ到着の前に髭でDetect)した瞬間1→ペグが到着した瞬間0になる線形減少関数。ペグが到着するまでのカウントダウン的な役割
        self.pegloc   : ペグの場所をそれぞれのペグごとに格納したリスト。新しくDetectされたペグはStepが進むごとに-1され、到着したタイミングで0になる。
        '''
        
        self.L_ang = 0
        self.R_ang = 0
        self.L_angV = 5.5*np.pi
        self.R_angV = 5.0*np.pi
        
        self.L_stimulus = 0
        self.R_stimulus = 0
        
        self.L_detect = 0
        self.R_detect = 0
        
        self.turntime = 0
        
        self.L_pegloc = np.array([-5])
        self.R_pegloc = np.array([-5])
        
        self.time = 0
        self.STEP = 0
        self.action = dict({"Left":1, "Right":1})
        
        obs = np.array([self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stimulus, self.L_detect, self.R_detect, self.turntime], dtype='float32')
#         obs = np.array([self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_detect, self.R_detect], dtype='float32')
#         obs = np.array([self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stimulus], dtype='float32')
        
        return obs

    def step(self, action_label):
        self.time += int(self.dt*1000)
        self.turntime = self.time % self.oneturn
        done = False
        
        ### 何もしなければどんどん角速度が減っていく
        self.L_angV  *= 0.98
        self.R_angV  *= 0.98
        
        ### 左右の脚の角速度を更新
        self.action["Left"] = action_label // self.action_num
        self.action["Right"] = action_label % self.action_num
        
        if self.action["Left"] == 0: # TORQUE UP SMALL
            self.L_angV += 0.75 
        elif self.action["Left"] == 2: # TORQUE UP LARGE
            self.L_angV += 1.5
            
        if self.action["Right"] == 0: #TORQUE UP SMALL
            self.R_angV += 0.75
        elif self.action["Right"] == 2: # TORQUE UP LARGE
            self.R_angV += 1.5
        
        ### 左右の脚の角度(位相)を更新
        dR_ang = self.dt * self.R_angV
        dL_ang = self.dt * self.L_angV
        
        self.R_ang += dR_ang
        self.L_ang += dL_ang
        
        ### Stimulationを更新
        self.L_pegloc = self.L_pegloc[self.L_pegloc>-4] ## ペグが到着してから4frame過ぎたらpeglocから消去する
        self.R_pegloc = self.R_pegloc[self.R_pegloc>-4] 
        if len(self.L_pegloc) != 0:
            for i in range(len(self.L_pegloc)):
                self.L_pegloc[i] -= 1
        if len(self.R_pegloc) != 0:
            for i in range(len(self.R_pegloc)):
                self.R_pegloc[i] -= 1
        
        L_upcoming = len(self.L_pegloc[np.where(self.L_pegloc)])
        R_upcoming = len(self.R_pegloc[np.where(self.R_pegloc)])
        
        self.L_stimulus -= L_upcoming * (1/self.frame)
        self.R_stimulus -= R_upcoming * (1/self.frame)
        
        if self.time%self.oneturn in self.Lpeg: # Peg Patternを元にDetectしたかどうかを判定
            self.L_detect = 1
            self.L_pegloc = np.append(self.L_pegloc, self.frame)
            self.L_stimulus += 1
        else:
            self.L_detect = 0
            
        if self.time%self.oneturn in self.Rdet:
            self.R_detect = 1
            self.R_pegloc = np.append(self.R_pegloc, self.frame)
            self.R_stimulus += 1
        else:
            self.R_detect = 0
        
        """
        int_value  : 周期の変化が少ない場合(ACTIONがNONE、つまり角加速度が0の場合)報酬をもらえる。
        phase_value: 位相の変化が少ない場合(右と左脚のAngular Velocityの差が小さい)報酬をもらえる。(angular velocityの逆数で報酬を与える)
        """
        
        int_value = 0
        phase_value = 0
        
        if self.action["Left"] in [0,1]: # SMALL or NONE
            int_value += 1.5
        if self.action["Right"] in [0,1]: # SMALL or NONE
            int_value += 1.5
        phase_value = 1/(0.1 + (self.R_angV - self.L_angV)**2)*3
        reward = int_value + phase_value
        
        
        ## スピードが遅すぎたら罰
        if self.R_angV < 2*np.pi:
            if self.print_failed_reason:
                print("not enough speed")
            done = True
            reward = -5
        else:
            reward += 2
        
        if self.L_angV < 2*np.pi and (not done):
            if self.print_failed_reason:
                print("not enough speed")
            done = True
            reward -= 5
        else:
            reward += 2
            
        if (self.R_ang > 2*np.pi) and (not done):
            self.STEP += 1
            if len(self.R_pegloc)==0:  ## 脚をついた時にそもそもPegをDetectしてなかった時は失敗
                self.R_ang -= 2*np.pi
                reward -= 5
                if self.print_failed_reason:
                    print("Touch failed")
                done = True
                
            elif self.R_pegloc[0] > 5: ##　脚をついたときにPeglocationが-5〜5の間 (-50ms〜50msの間)ならばOK、それ以外(>5)なら失敗(ペグの場所に脚をつけなかった。)
                self.R_ang -= 2*np.pi
                reward -= 5
                if self.print_failed_reason:
                    print("Touch failed")
                done = True
                
            elif self.STEP >= self.maxSTEP: ## Max歩以上走れたときは成功、報酬を与えて終了
                reward += 100
                done = True
                
            else:
                self.R_ang -= 2*np.pi ## 普通に成功したときはrewardをちょっとだけ与えて続行
                reward += 100
                self.STEP += 1
        
        if (self.L_ang > 2*np.pi) and (not done):
            self.STEP += 1
            if len(self.L_pegloc)==0:  ## 脚をついた時にそもそもPegをDetectしてなかった時は失敗
                self.L_ang -= 2*np.pi
                reward -= 5
                if self.print_failed_reason:
                    print("Touch failed")
                done = True
                
            elif self.L_pegloc[0] > 5: ##　脚をついたときにPeglocationが-5〜5の間 (-50ms〜50msの間)ならばOK、それ以外なら失敗
                self.L_ang -= 2*np.pi
                reward -= 5
                if self.print_failed_reason:
                    print("Touch failed")
                done = True
                
            elif self.STEP >= self.maxSTEP: ## max歩以上走れたときは成功、報酬を与えて終了
                reward += 100
                done = True
                
            else:
                self.L_ang -= 2*np.pi ## 普通に成功したときはrewardをちょっとだけ与えて続行
                reward += 100
                self.STEP += 1
                
        if done:
            if self.print_failed_reason:
                print(reward)
            assert -5 <= reward
        obs = np.array([self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stimulus, self.L_detect, self.R_detect, self.turntime], dtype='float32')
#         obs = np.array([self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_detect, self.R_detect], dtype='float32')
#         obs = np.array([self.L_ang, self.R_ang, self.L_angV, self.R_angV, self.L_stimulus, self.R_stimulus], dtype='float32')
        
        info = dict(
            {"dR_ang":dR_ang,
             "dL_ang":dL_ang,
             "int_value":int_value,
             "phase_value":phase_value}
        )
        
        return obs, reward, done, info

    def render(self, mode='human'):
        time.sleep(0.1)
        img = np.full((1200, 1200, 3), 255, dtype="int16") #画面初期化
        
        img = cv2.circle(img,center=(400,300), radius=200, color=(127,127,127), thickness=10)
        img = cv2.circle(img,center=(400,900), radius=200, color=(127,127,127), thickness=10)
        
        img = cv2.line(img, # Left Cycle
                       pt1=(400,300),
                       pt2=(int(400-np.sin(self.L_ang)*200), int(300+np.cos(self.L_ang)*200)),
                       color=(255, 0, 0),
                       thickness=10
                      )
        img = cv2.line(img, # Right Cycle
                       pt1=(400,900),
                       pt2=(int(400-np.sin(self.R_ang)*200), int(900+np.cos(self.R_ang)*200)),
                       color=(0, 0, 255),
                       thickness=10
                      )
        
        if len(self.L_pegloc) != 0:
            for i in range(len(self.L_pegloc)):
                img = cv2.line(img,
                               pt1=(int(300+self.L_pegloc[i]*40), 500),
                               pt2=(int(500+self.L_pegloc[i]*40), 500),
                               color=(0, 0, 0),
                               thickness=10
                              )
        if len(self.R_pegloc) != 0:
            for i in range(len(self.R_pegloc)):
                img = cv2.line(img,
                               pt1=(int(300+self.R_pegloc[i]*40), 1100),
                               pt2=(int(500+self.R_pegloc[i]*40), 1100),
                               color=(0, 0, 0),
                               thickness=10
                              )

        img = cv2.putText(img,text="Left Cycle",org=(240,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,0,0),thickness=8)
        img = cv2.putText(img,text="Right Cycle",org=(240,660),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,0,0),thickness=8)

        self.im.set_array(img)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()