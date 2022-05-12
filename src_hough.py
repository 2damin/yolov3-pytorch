# -*- coding: utf-8 -*-
"""
Created on Thu May 12 05:13:00 2022

@author: Milly
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from xycar_msgs.msg import xycar_motor
from std_msgs.msg import String
from yolov3_trt.msg import BoundingBoxes, BoundingBox
import sys
import os
import signal


frame = np.empty(shape = [0])
bridge = CvBridge()
pub = None

Width = 640
Height = 480
Offset = 380
Gap = 55
speed  = 15
angle = 0
count = 0
prev_speed = 0
prev_lpos = 0
prev_rpos = 0
Purpose_speed = 15

Kp = 0.35
Ki = 0
Kd = 0.15

p_error, d_error, i_error = 0,0,0

obj_id = -1
box_width = 0
box_height = 0

bbox_threshold = "정해야 함"

Stopped = False # 정지, 횡단보도 표지판이 있을 때 멈췄다가 5초뒤 출발해야 함.
                # 5초간 정지했으면 이미지에 해당 표지판이 있더라도 무시하고 주행해야함.
'''
BoundingBox.msg

float64 probability
int64 xmin
int64 ymin
int64 xmax
int64 ymax
int16 id
'''

def box_callback(data):
    global obj_id, Stopped
    for bbox in data.bounding_boxes:
        curr_box_width = bbox.xmax-bbox.xmin
        curr_box_height = bbox.ymax=bbox.ymin
        if (curr_box_width*curr_box_height > bbox_threshold) and (curr_box_width*curr_box_height > box_width*box_height):
            obj_id = bbox.id

    #멈춰야하는 표지판을 지난 후, 다시 정지표시판 만났을 때를 위해 변경
    if (Stopped and (obj_id not in [2,4])) :
        Stopped = False
        
    
def PID_control(error) :
    global Kp, Kd, Ki
    global p_error, d_error, i_error

    d_error = error - p_error
    p_error = error
    i_error += error
    return Kp * p_error + Kd * d_error + Ki * i_error

def img_callback(data) :
    global frame
    frame = bridge.imgmsg_to_cv2(data, "bgr8")


# 인식 된 직선을 왼,오른 차선으로 구분
def divide_left_right(lines):
    min_slope_threshold = 0
    max_slope_threshold = 10

    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)

				#차선이 아닌 직선 필터링 (ex.수평선)
        if abs(slope) > min_slope_threshold and abs(slope) < max_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    left_lines = []
    right_lines = []
    
    for i in range(len(slopes)):
        slope = slopes[i]
        x1, y1, x2, y2 = new_lines[i]

        if (slope < 0) and (x2 < Width / 2):
            left_lines.append([new_lines[i].tolist()])
        elif (slope > 0) and (x1 > Width / 2):
            right_lines.append([new_lines[i].tolist()])

    return left_lines, right_lines

# 차선의 대표 직선을 찾고, 기울기, 절편을 반환   
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    num = len(lines)
    if num == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (num * 2)
    y_avg = y_sum / (num * 2)
    m = m_sum / num
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):

    m, b = get_line_params(lines)

#차선을 못찾았을 경우 왼쪽은 -1, 오른쪽은 641로함.
    if m == 0 and b == 0:
        if left:
            pos = -1
        if right:
            pos = Width + 1
    else:
        y = Gap / 2 
        pos = (y - b) / m

    return int(pos)

# 영상 전처리 및 차선 찾기
def process_image(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    norm = cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)        
    blur_gray = cv2.GaussianBlur(norm, (5, 5), 0)
        
    edge_img = cv2.Canny(np.uint8(blur_gray), 140, 70)

    roi = edge_img[Offset: Offset + Gap, 0: Width]
        
    all_lines = cv2.HoughLinesP(roi, 1, math.pi / 180, 30, 30, 10)

    if all_lines is None:
        return 0, 640
    left_lines, right_lines = divide_left_right(all_lines)

    lpos = get_line_pos(frame, left_lines, left=True)
    rpos = get_line_pos(frame, right_lines, right=True)
    return lpos, rpos

#publish xycar_motor msg
def drive(Angle, Speed):
    
    msg = xycar_motor()
        
    if Angle >= 50:
        Angle = 50
    if Angle <= -50:
        Angle = -50
            
    msg.angle = Angle
        
    #제어 안정성을 높이기 위해 천천히 가속
    if Speed != -10:
        if Speed < Purpose_speed:
            Speed += 1 
        else:
            Speed = Purpose_speed
    
		#곡선 주행시 차선을 이탈하지 않도록 각도에 비례해 감속            
    if abs(Angle) > 20 and Speed != -10:
        msg.speed = (Speed -(0.17 * abs(Angle)))
    else:
        msg.speed = (Speed -(0.1 * abs(Angle)))

    pub.publish(msg)

# 차선 잃을시 멈췄다 후진
'''
속도 느려서 아마 삭제해도 될거같음. 이거 때문에 문제 중앙 교차로에서 문제될듯.
stop_back안해도 차선인식 기반 주행에서 문제안되면 빼자.

def stop_back():
    global speed, angle
    
    rate = rospy.Rate(10)
		#급하게 속도를 바꿔서 오히려 더 차선을 잃지 않게 천천히 감속
    while True:
        if speed>=0:
            speed -= 5
            drive(angle,speed)
        else:
            break
    speed = -10
    
		#0.5초간 반대 조향각으로 후진
    for i in range(5):
        drive(-angle/5, speed)
        rate.sleep()
    speed = 1 
'''       
    
def start():
    
    global Stopped
    global Width, Height, cap, prev_l, prev_r
    global pub, frame, count, angle, speed

    rospy.init_node("auto_drive")
    pub = rospy.Publisher('xycar_motor',xycar_motor,queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,img_callback)
    bbox_sub = rospy.Subscriber("/yolov3_trt_ros/detections", BoundingBoxes, box_callback, queue_size=1)
    rospy.sleep(2)
    
    drive(0,0)
        
    while True:
        while not frame.size == (Width*Height*3) :
            continue
        
        #라벨링은 신호등으로 통일 되어있는데, 초록색, 빨강색으로 분리해야할듯
        '''
        if traffic_red:
            drive(angle,0)
            continue
        '''    
        # 정지, 횡단보도 만나면 그대로 6초간 정지
        if (not Stopped) and (obj_id in [2,4]) :
            drive(angle,0)
            rospy.sleep(6.)
            Stopped = True
            continue
        
        lpos, rpos = process_image(frame)
        
        #좌,우회전 갈림길에서 한쪽차선은 비어있음.
        if obj_id == 0:
            rpos = Width + 1
        elif obj_id == 1:
            lpos = -1
				
		#차선 한쪽을 잃었을 경우 440 픽셀 옆에 있다고 가정
        if lpos == -1 and rpos != Width + 1:
            lpos = rpos - 440

        elif rpos == Width + 1 and lpos != -1:
            rpos = lpos + 440
				
		#차선을 둘다 놓치는 중앙교차로에서 원래 angle, speed 유지해서 주행.
        #원래는 차선 놓치면 후진 시켰지만 일단 시범주행해보고 다시 넣을지 말지 결정
        elif rpos == Width + 1 and lpos == -1:
            drive(angle,speed)
            continue

		#도로 내부에 다른 물체때문에 오검출 되었을 경우 처리
        else:
            if rpos - lpos < 430:
                if abs(rpos-prev_rpos) < 100:
                    lpos = rpos - 450 
                if abs(lpos-prev_lpos) < 100:
                    rpos = lpos + 450 

        
        center = (lpos + rpos) / 2 
        error = (center - (Width / 2))

        angle = PID_control(error)

        drive(angle , speed)
        prev_l = lpos
        prev_r = rpos
        
        
    rospy.spin()
    
if __name__ == '__main__':
    start()
