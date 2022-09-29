import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

fig = plt.figure()
ax = fig.add_subplot(111)
img = np.full([1200, 1200, 3], 255, dtype="int16")
im = ax.imshow(img)
plt.show(block=False)
i = 0
while True:
    i+=1
    time.sleep(0.01)
    L_ang = i*0.2
    R_ang = i*0.25
    img = np.full((1200, 1200, 3), 255, dtype="int16") #画面初期化
    
    img = cv2.line(img, pt1=(400,300), pt2=(int(400+np.cos(L_ang)*200), int(300+np.sin(L_ang)*200)), color=(255, 0, 0), thickness=10)
    img = cv2.circle(img, center=(400,300), radius=200, color=(127,127,127), thickness=10)
    img = cv2.line(img, pt1=(400,900), pt2=(int(400+np.cos(R_ang)*200), int(900+np.sin(R_ang)*200)), color=(0, 0, 255), thickness=10)
    img = cv2.circle(img, center=(400,900), radius=200, color=(127,127,127), thickness=10)

    img = cv2.line(img, pt1=(1100-i*10,500), pt2=(900-i*10,500), color=(0, 0, 0), thickness=10)
    img = cv2.line(img, pt1=(900-i*10,1100), pt2=(700-i*10,1100), color=(0, 0, 0), thickness=10)

    img = cv2.putText(img,text="Left Cycle",org=(240,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,0,0),thickness=8)
    img = cv2.putText(img,text="Right Cycle",org=(240,660),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,0,0),thickness=8)

    im.set_array(img)
    fig.canvas.draw()
    fig.canvas.flush_events()
