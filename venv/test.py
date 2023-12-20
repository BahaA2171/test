from moviepy.editor import VideoFileClip
import cv2
import PySimpleGUI as sg
import os
import sys
from datetime  import datetime
from pathlib import Path
import mediapipe as mp
import handmodel as htm
import time
import math
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#--------------------variables--------------------

m=0
o=0
tab_blink,tab_hand,tab_pose,tab_move,all=[],[],[],[],[]
t=True
indx=0
ex=0
test=1
pTime = 0
pt=0
dt=time.time()
i_eye=0
x1,y1,x2,y2=10,10,0,0
#------------------------functions--------------------------
def update_figure(data,i=0):
    axes = fig.axes
    l=['y-','r-','b-','g-','r-']
    x=[i[1] for i in data]
    y=[i[0] for i in data]
    axes[i].plot(x,y,l[i])
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack()

def getContours(img,imgc,ex):
    contours,h=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x=ex
    for cnt in contours :
        area = cv2.contourArea(cnt)

        if area>20 :
              #print(area)
              cv2.drawContours(imgc, cnt, -1, (0, 255, 255), 3)
              p=cv2.arcLength(cnt,True)
              approx= cv2.approxPolyDP(cnt,0.02*p,True)
              #print(approx)
              xl=len(approx)
              x, y, h, z = cv2.boundingRect(approx)
              #print(x)
    return x
#_______________path_function____________#
def resource_path(relative_path):
    base_path = getattr(
        sys,
        '_MEIPASS',
        os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
#---------------------face-----------------------------------
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

#--------------------hand_detector----------------------------------
detector = htm.handDetector(maxHands=1)

#------------------------pose-------------------------------
mpPose = mp.solutions.pose
pose = mpPose.Pose()
#------------------------function--------------------------
def play(path):
    return cv2.VideoCapture(path)

#------------------------gui--------------------------------
menu_layout=[["play",["video","cam"]]
             ]
#----screen1--------#
def create_window(theme):
    sg.theme(theme)
    layout=[[sg.Menu(menu_layout)],
            [sg.Canvas(key='-CANVAS-'),sg.Image('background2.png', key="img", pad=10)],
             ]
    return sg.Window("EI", layout, grab_anywhere=True,  element_justification='c',finalize=True)


theme='LightGray1'
window=create_window(theme)

fig = matplotlib.figure.Figure(figsize = (5,4))
ax1=fig.add_subplot(321,title="eye blinks").plot([],[])
ax2=fig.add_subplot(322,title="hand").plot([],[])
ax3=fig.add_subplot(324,title="pose").plot([],[])
ax4=fig.add_subplot(323,title="eye_move").plot([],[])
ax5=fig.add_subplot(325,title="stress").plot([],[])

figure_canvas_agg = FigureCanvasTkAgg(fig,window['-CANVAS-'].TKCanvas)
figure_canvas_agg.draw()
figure_canvas_agg.get_tk_widget().pack()



#-----------video/cam------------
file=0
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
   event,valeus=window.read(timeout=10)
   success, img = cap.read()
   if not success:
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       cap = cv2.VideoCapture(str(file))
       success, img = cap.read()
       continue
   img=cv2.flip(img,2)

   ih, iw, ic = img.shape
   mask = np.zeros((ih, iw), dtype=np.uint8)
   imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   results = faceMesh.process(imgRGB)
   if results.multi_face_landmarks:
    for faceLms in results.multi_face_landmarks:
       #mpDraw.draw_landmarks(img, faceLms)
       right_eye=((faceLms.landmark[159].x* iw, faceLms.landmark[159].y*ih-10),(faceLms.landmark[157].x* iw,faceLms.landmark[157].y*ih-10)
                  ,(faceLms.landmark[133].x* iw+10, faceLms.landmark[133].y*ih),(faceLms.landmark[145].x* iw,faceLms.landmark[145].y*ih+10)
                  ,(faceLms.landmark[33].x* iw-10,faceLms.landmark[33].y*ih), (faceLms.landmark[161].x* iw,faceLms.landmark[161].y*ih-10),
                  )
       cv2.fillPoly(mask,[np.array(right_eye, dtype=np.int32)],(255,255,255))
#--------------------------pose--------------------------------
       results = pose.process(imgRGB)
       if results.pose_landmarks:
           #mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
           #print(results.pose_landmarks.landmark)
           pr1=[int(results.pose_landmarks.landmark[14].x * iw),int(results.pose_landmarks.landmark[14].y * ih)]
           pl1 = [int(results.pose_landmarks.landmark[13].x * iw), int(results.pose_landmarks.landmark[13].y * ih)]
           pr2 = [int(results.pose_landmarks.landmark[12].x * iw), int(results.pose_landmarks.landmark[12].y * ih)]
           pl2 = [int(results.pose_landmarks.landmark[11].x * iw), int(results.pose_landmarks.landmark[11].y * ih)]
           cv2.circle(img, pr1, 10, (255, 0, 255), cv2.FILLED)
           cv2.circle(img, pl1, 10, (255, 0, 255), cv2.FILLED)
           cv2.circle(img, pr2, 10, (255, 0, 255), cv2.FILLED)
           cv2.circle(img, pl2, 10, (255, 0, 255), cv2.FILLED)
           cv2.line(img, pr1, pl1, (0, 0, 255), 3)
           d1=math.hypot(pl1[0] - pr1[0], pl1[1] - pr1[1])
           cv2.putText(img,str(int(d1)), ((pr1[0]+pl1[0])//2,(pr1[1]+pl1[1])//2), cv2.FONT_HERSHEY_PLAIN, 1, (255,0 , 255), 1)
           cv2.line(img, pr2, pl2, (0, 0, 255), 3)
           d2 = math.hypot(pl2[0] - pr2[0], pl2[1] - pr2[1])
           cv2.putText(img, str(int(d2)), ((pr2[0] + pl2[0]) // 2, (pr2[1] + pl2[1]) // 2), cv2.FONT_HERSHEY_PLAIN, 1,
                       (255, 0, 255), 1)
           if d2<400 or d1<400:
               o=1
           if d2<300 or d2<300:
               o=2
           if d2<300 and d1<300:
               o=3




#---------------------------eye-----------------------------
       x1, y1 = faceLms.landmark[159].x, faceLms.landmark[159].y
       x1, y1 = int(x1 * iw), int(y1 * ih)
       if ex>x1 :
           m+=1
           cv2.putText(img, "eye_move", (x2, y2 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
           tab_move.append([m, int(pt - dt)])
           update_figure(tab_move, 3)
       x2, y2 = faceLms.landmark[145].x, faceLms.landmark[145].y
       x2, y2 = int(x2 * iw), int(y2 * ih)
      # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
       distance = math.hypot(x2 - x1, y2 - y1)

       if dt + 160 > pt:
           pt = time.time()
           if distance < 8 and test == 1:
               test = 0
               i_eye += 1
               tab_blink.append([i_eye,int(pt-dt)])
               update_figure(tab_blink)
           if distance > 10 and test == 0:
               test = 1

       cv2.putText(img, str(i_eye), (x2, y2 - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
       if dt + 160 < pt:
           dt = time.time()
           i_eye = 0
           indx = 0
           tab_hand=[]
           tab_blink=[]

           #-------------------------------cv2--------------------------------------
   img_g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   eye = cv2.bitwise_and(img_g,img_g, mask=mask)
   lower = np.array([0, 0,0 ])
   upper = np.array([115,115, 33])
   mask2 = cv2.inRange(img,lower, upper)
   eye = cv2.bitwise_and(eye, eye, mask=mask2)
   eye_blur = cv2.GaussianBlur(eye, (7, 7), 1)
   eye_canny = cv2.Canny(eye_blur, 50, 50)
   #cv2.imshow('f',eye_canny)
   ex=getContours(eye_canny,img,0)
   #cv2.imshow("e",eye)
   cTime = time.time()
   fps = 1 / (cTime - pTime)
   pTime = cTime
   #cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
              # 3, (255, 0, 0), 1)
#----------------------------hand---------------------------------------
   tab_pose.append([o, int(pt - dt)])
   update_figure(tab_pose, 2)
   img = detector.findHands(img,draw=False)
   list = detector.findPosition(img, draw=False)
   if len(list) != 0:
       h1x, h1y = list[8][1:]
       h2x, h2y = list[12][1:]
       h3x, h3y = list[11][1:]
       if(h1y<h2y):
           if (h1y<h3y)and t==True:
               indx+=1
               tab_hand.append([indx, int(pt - dt)])
               update_figure(tab_hand,1)
               t = False
               print(indx)
           if (h1y>h3y) and t==False:
               t = True
       cv2.circle(img, (h1x,h1y), 10, (255, 0, 255), cv2.FILLED)
       cv2.circle(img, (h2x,h2y), 10, (255, 0, 255), cv2.FILLED)
       cv2.circle(img, (h3x,h3y), 10, (255, 0, 255), cv2.FILLED)



   imgbytes = cv2.imencode('.png', img)[1].tobytes()
   window["img"].update(data=imgbytes)
#-------------------------------------------------------------------------
   stress=o*0.25+indx*0.05+i_eye*0.15+m*0.15
   all.append([stress, int(pt - dt)])
   update_figure(all,4)

#------------------------------------------------------------------
   if event == "video":
       file_path = sg.popup_get_file("open", no_window=True)
       if file_path:
           file = Path(file_path)
           f = file_path.split("/")
           cap = play(str(file))
   if event == "cam":
       cap=play(0)

window.close()