import cv2
import numpy as np
import matplotlib.pyplot as plt

while True:
    i = np.random.randint(0,100)
    img = np.full([1200, 1200, 3], 255, dtype="int16") #画面初期化
    
    img = cv2.line(img,pt1=(400,300),pt2=(400,500),color=(255, 0, 0),thickness=10)
    img = cv2.circle(img,center=(400,300), radius=200, color=(127,127,127), thickness=10)
    img = cv2.line(img,pt1=(400,900),pt2=(400,1100),color=(0, 0, 255),thickness=10)
    img = cv2.circle(img,center=(400,900), radius=200, color=(127,127,127), thickness=10)

    img = cv2.line(img,pt1=(1100-i,500),pt2=(900-i,500),color=(0, 0, 0),thickness=10)
    img = cv2.line(img,pt1=(900-i,1100),pt2=(700-i,1100),color=(0, 0, 0),thickness=10)

    img = cv2.putText(img,text="Left Cycle",org=(240,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,0,0),thickness=8)
    img = cv2.putText(img,text="Right Cycle",org=(240,660),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,0,0),thickness=8)
    
    cv2.imshow("Image", img)
    cv2.waitKey(20)
    
cv2.destroyAllWindows()