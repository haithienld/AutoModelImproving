#!/usr/bin/python3
# -*- coding: utf-8 -*-

''' run with image 
python3 6_multicam_soha.py --image enabled  --image_in samples/298.jpg --image_out sample/222.jpg  --horizontal_ratio 0.7 --vertical_ratio 0.7
run with video or webcam 
python3 6_multicam_soha.py --video enabled --stream_in samples/stream_in.mp4 --stream_out samples/aa.mp4 --horizontal_ratio 0.7 --vertical_ratio 0.7

Run Jupyter:
Coral: jupyter notebook   --NotebookApp.allow_origin='https://colab.research.google.com'   --port=8888   --NotebookApp.port_retries=0
jupyter_http_over_ws extension initialized. Listening on /http_over_websocket
server :ssh -N -L 8888:localhost:8888 mendel@192.168.100.2 -i ~/.config/mdt/keys/mdt.key

install tflite_runtime
sudo apt-get install python3-edgetpu
https://google-coral.github.io/py-repo/
tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
'''
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
##
import xgboost as xgb
import argparse
import itertools
import collections
from functools import partial
import re
import time

import numpy as np
from PIL import Image
import svgwrite
import gstreamer
#####
import sys
import base64
import zmq
from zmq import ssh
import common
import cv2
import os
import math
#####
from pose_engine import PoseEngine
from pose_engine import KeypointType
from datetime import datetime

########## Human Action Lib ##############
import pickle
import multiprocessing
#import pandas as pd
#import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#===========streamming======================
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://192.168.0.6:4664') #host - the host to view camera
#ssh.tunnel_connection(footage_socket,'tcp://192.168.0.6:4664',"tommy@147.47.200.65:12345")


#footage_socket.connect('tcp://147.47.200.65:12345;192.168.0.6:4664 ') #host - the host to view camera
#===========================================

class SocialDistancing:
    
    colors = [(0, 255, 0), (0, 0, 255)]

    nd_color = [(153, 0, 51), (153, 0, 0),
                (153, 51, 0), (153, 102, 0),
                (153, 153, 0), (102, 153, 0),
                (51, 153, 0), (0, 153, 0),
                (0, 102, 153), (0, 153, 51),
                (0, 153, 102), (0, 153, 153),
                (0, 102, 153), (0, 51, 153),
                (0, 0, 153), (153, 0, 102),
                (102, 0, 153), (153, 0, 153),
                (102, 0, 153), (0, 0, 153),
                (0, 0, 153)
                ]
    connections = [(0, 1), (0, 2), (0, 3), (0, 4),
                   (3, 1), (4, 2), (1, 2), (5, 6),
                   (5, 7), (5, 11), (6, 8), (6, 12), 
                   (7, 9), (8, 10), (11, 12), (11, 13),
                   (12, 14), (13, 15), (14, 16)]

    EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
    )
    '''
    EDGES = (
    ('nose', 'left eye'), #0 1
    ('nose', 'right eye'), #0 2 
    ('nose', 'left ear'), #0 3
    ('nose', 'right ear'), #0 4
    ('left ear', 'left eye'), #3 1 
    ('right ear', 'right eye'), #4 2 
    ('left eye', 'right eye'), #1 2 
    ('left shoulder', 'right shoulder'), #5 6
    ('left shoulder', 'left elbow'), #5 7 
    ('left shoulder', 'left hip'), #5 11
    ('right shoulder', 'right elbow'), #6 8
    ('right shoulder', 'right hip'), #6 12
    ('left elbow', 'left wrist'), #7 9
    ('right elbow', 'right wrist'), # 8 10
    ('left hip', 'right hip'),# 11 12
    ('left hip', 'left knee'), # 11 13
    ('right hip', 'right knee'), # 12 14
    ('left knee', 'left ankle'), # 13 15
    ('right knee', 'right ankle'), # 14 16    
    )
    
    0	nose
    1	leftEye
    2	rightEye
    3	leftEar
    4	rightEar
    5	leftShoulder
    6	rightShoulder
    7	leftElbow
    8	rightElbow
    9	leftWrist
    10	rightWrist
    11	leftHip
    12	rightHip
    13	leftKnee
    14	rightKnee
    15	leftAnkle
    16	rightAnkle
    '''
    '''
        Initialize Object
    '''

    def __init__(self, args):
        # Ratio params
        self.engine = PoseEngine(args[0].model)
        
        horizontal_ratio = float(args[0].horizontal_ratio)
        vertical_ratio = float(args[0].vertical_ratio)

        # Check video
        if args[0].video != "enabled" and args[0].video != "disabled":
            print("Error: set correct video mode, enabled or disabled", flush=True)
            sys.exit(-1)

        # Check video
        if args[0].image != "enabled" and args[0].image != "disabled":
            print("Error: set correct image mode, enabled or disabled", flush=True)
            sys.exit(-1)

        # Convert args to boolean
        self.use_video = True if args[0].video == "enabled" else False

        self.use_image = True if args[0].image == "enabled" else False
       
        # Unable to use video and image mode at same time
        if self.use_video and self.use_image:
            print("Error: unable to use video and image mode at the same time!", flush=True)
            sys.exit(-1)

        # Unable to not use or video or image mode at same time
        if self.use_video and self.use_image:
            print("Error: enable or video or image mode!", flush=True)
            sys.exit(-1)
        
        if self.use_video:
            # Open video capture
            print(args[0].stream_in)
            if(args[0].stream_in=="0"):
                index = int(args[0].stream_in)
                self.cap = cv2.VideoCapture(index) #cv2.VideoCapture(1)
            else:
                self.cap = cv2.VideoCapture(args[0].stream_in) #cv2.VideoCapture(1)

            if not self.cap.isOpened():
                print("Error: Opening video stream or file {0}".format(
                    args[0].stream_in), flush=True)
                sys.exit(-1)

            # Get input size
            width = int(self.cap.get(3))
            height = int(self.cap.get(4))
         
            # Get image size
            im_size = (width, height)
            
            #Write image
            self.out = cv2.VideoWriter(args[0].stream_out, cv2.VideoWriter_fourcc(*args[0].encoding_codec),
                                           int(self.cap.get(cv2.CAP_PROP_FPS)), (640, 480))

            if self.out is None:
                print("Error: Unable to open output video file {0}".format(
                    args[0].stream_out), flush=True)
                sys.exit(-1)

        if self.use_image:
            self.image = cv2.imread(args[0].image_in)
            if self.image is None:
                print("Error: Unable to open input image file {0}".format(
                    args[0].image_in), flush=True)
                sys.exit(-1)

            self.image_out = args[0].image_out

            # Get image size
            im_size = (self.image.shape[1], self.image.shape[0])

        # Compute Homograpy
        self.homography_matrix = self.compute_homography(
            horizontal_ratio, vertical_ratio, im_size)

        self.background_masked = False
        # Open image backgrouns, if it is necessary
        if args[0].masked == "enabled":
            # Set masked flag
            self.background_masked = True

            # Load static background
            self.background_image = cv2.imread(args[0].background_in)

            # Close, if no background, but required
            if self.background_image is None:
                print("Error: Unable to load background image (flag enabled)", flush=True)
                sys.exit(-1)
 
        # Calibrate heigh value
        self.calibrate = float(args[0].calibration)

        # Actually unused
        self.ellipse_angle = 0

        # Body confidence threshold
        self.body_th = float(args[0].body_threshold)

        # Show confidence
        self.show_confidence = True if args[0].show_confidence == "enabled" else False

        # Set mask precision (mask division factor)
        self.overlap_precision = int(args[0].overlap_precision)

        # Check user value
        self.overlap_precision = 16 if self.overlap_precision > 16 else self.overlap_precision

        self.overlap_precision = 1 if self.overlap_precision < 0 else self.overlap_precision
      

    '''
        Draw Skelethon
    '''

    def draw_skeleton(self, frame, keypoints, colour):

        for keypoint_id1, keypoint_id2 in self.connections:
            x1, y1 = keypoints[keypoint_id1]
            x2, y2 = keypoints[keypoint_id2]

            if 0 in (x1, y1, x2, y2):
                continue

            pt1 = int(round(x1)), int(round(y1))
            pt2 = int(round(x2)), int(round(y2))

            cv2.circle(frame, center=pt1, radius=4,
                       color=self.nd_color[keypoint_id2], thickness=-1)
            cv2.line(frame, pt1=pt1, pt2=pt2,
                     color=self.nd_color[keypoint_id2], thickness=2)

    '''
        Compute skelethon bounding box
    '''

    def compute_simple_bounding_box(self, skeleton):
        x = skeleton[::2]
        x = np.where(x == 0.0, np.nan, x)
        left, right = int(round(np.nanmin(x))), int(round(np.nanmax(x)))
        y = skeleton[1::2]
        y = np.where(y == 0.0, np.nan, y)
        top, bottom = int(round(np.nanmin(y))), int(round(np.nanmax(y)))
        return left, right, top, bottom

    '''
        Compute Homograpy
    '''

    def compute_homography(self, H_ratio, V_ratio, im_size):
        rationed_hight = im_size[1] * V_ratio
        rationed_width = im_size[0] * H_ratio
        src = np.array([[0, 0], [0, im_size[1]], [
            im_size[0], im_size[1]], [im_size[0], 0]])
        dst = np.array([[0+rationed_width/2, 0+rationed_hight], [0, im_size[1]], [im_size[0],
                                                                                  im_size[1]], [im_size[0]-rationed_width/2, 0+rationed_hight]], np.int32)
        h, status = cv2.findHomography(src, dst)
        return h

    '''
        Compute overlap
    '''

    def compute_overlap(self, rect_1, rect_2):
        x_overlap = max(
            0, min(rect_1[1], rect_2[1]) - max(rect_1[0], rect_2[0]))
        y_overlap = max(
            0, min(rect_1[3], rect_2[3]) - max(rect_1[2], rect_2[2]))
        overlapArea = x_overlap * y_overlap
        if overlapArea:
            overlaps = True
        else:
            overlaps = False
        return overlaps

    
    
    '''
        Trace results
    '''

    def trace(self, image, skeletal_coordinates, draw_ellipse_requirements, is_skeletal_overlapped,show_dis_from_cam):
        bodys = []

        # Trace ellipses and body on target image
        i = 0

        for skeletal_coordinate in skeletal_coordinates[0]:
            if float(skeletal_coordinates[1][i]) < self.body_th:
                continue

            # Trace ellipse
            cv2.ellipse(image,
                        (int(draw_ellipse_requirements[i][0]), int(
                            draw_ellipse_requirements[i][1])),
                        (int(draw_ellipse_requirements[i][2]), int(
                            draw_ellipse_requirements[i][3])), 0, 0, 360,
                        self.colors[int(is_skeletal_overlapped[i])], 3)
            #Trace dis from cam
            print("show_dis_from_cam",show_dis_from_cam)
            cv2.putText(image, ""+str(show_dis_from_cam[i][0])+"", (int(show_dis_from_cam[i][1]),int(show_dis_from_cam[i][2])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255,0), 2)
            cv2.putText(image, ""+str(skeletal_coordinates[2][i])+"", (int(show_dis_from_cam[i][1]),int(show_dis_from_cam[i][2])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255,0), 2)

            #==================================Display Welcome Customer===============================
            if(str(skeletal_coordinates[2][i])=="['standing']"):
                text_line = 'Welcome to GSDS'
            #text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f' % (avg_inference_time, 1000 / avg_inference_time, next(fps_counter)) 
            #cv2_im = cv2.rectangle(cv2_im, (0), 0), (x1, y1), (0, 255, 0), 2)
            #cv2.putText(image,text_line, (100,100), 0, 2, 20)
                cv2.putText(image, text_line , (0,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0,0), 2)
            else:
                cv2.putText(image, "BAB" , (0,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0,0), 2)    
            #======================================================================================
            # Trace skelethon
            skeletal_coordinate = np.array(skeletal_coordinate)
            
            self.draw_skeleton(
                image, skeletal_coordinate.reshape(-1, 2), (255, 0, 0))

            if int(skeletal_coordinate[2]) != 0 and int(skeletal_coordinate[3]) != 0 and self.show_confidence:
                cv2.putText(image, "{0:.2f}".format(skeletal_coordinates[1][i]),
                            (int(skeletal_coordinate[2]), int(skeletal_coordinate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Append json body data, joints coordinates, ground ellipses
            bodys.append([[round(x) for x in skeletal_coordinate],
                          draw_ellipse_requirements[i], int(is_skeletal_overlapped[i])])

            i += 1
            
        #============streamming to server==============
        #img = np.hstack((cv2_im, cv2_sodidi))
        #thay frame = img
        
        encoded, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)
        
            
        #=============================================
            


    '''
        Evaluate skelethon height
    '''

    def evaluate_height(self, skeletal_coordinate):
        # Calculate skeleton height
        calculated_height = 0
        pointer = -1

        # Left leg
        joint_set = [12, 14, 16]

        # Check if leg is complete
        left_leg = True
        for k in joint_set:
            x = int(skeletal_coordinate[k*2])
            y = int(skeletal_coordinate[k*2+1])
            if x == 0 or y == 0:
                # No left leg, try right_leg
                joint_set = [9, 10, 11]
                left_leg = False
                break

        if not left_leg:
            joint_set = [11, 13, 15]
            # Check if leg is complete
            for k in joint_set:
                x = int(skeletal_coordinate[k*2])
                y = int(skeletal_coordinate[k*2+1])
                if x == 0 or y == 0:
                    # No left leg, no right leg, then body
                    joint_set = [0, 1, 8]
                    break

        # Evaluate leg height
        pointer = -1
        for k in joint_set[:-1]:
            pointer += 1
            if skeletal_coordinate[joint_set[pointer]*2]\
                    and skeletal_coordinate[joint_set[pointer+1]*2]\
                    and skeletal_coordinate[joint_set[pointer]*2+1]\
                    and skeletal_coordinate[joint_set[pointer+1]*2+1]:
                calculated_height = calculated_height +\
                    math.sqrt(((skeletal_coordinate[joint_set[pointer]*2] -
                                skeletal_coordinate[joint_set[pointer+1]*2])**2) +
                              ((skeletal_coordinate[joint_set[pointer]*2+1] -
                                skeletal_coordinate[joint_set[pointer+1]*2+1])**2))

        # Set parameter (calibrate) to optimize settings (camera dependent)
        return calculated_height * self.calibrate

    '''
        Evaluate overlapping
    '''

    def evaluate_overlapping(self, ellipse_boxes, is_skeletal_overlapped, ellipse_pool):
        # checks for overlaps between people's ellipses, to determine risky or not
        for ind1, ind2 in itertools.combinations(list(range(0, len(ellipse_pool))), 2):

            is_overlap = cv2.bitwise_and(
                ellipse_pool[ind1], ellipse_pool[ind2])

            if is_overlap.any() and (not is_skeletal_overlapped[ind1] or not is_skeletal_overlapped[ind2]):
                is_skeletal_overlapped[ind1] = 1
                is_skeletal_overlapped[ind2] = 1

    '''
        Create Joint Array
    '''

    def create_joint_array(self, skeletal_coordinates,inference_box,src_size,xgb_model_loaded):
        # Get joints sequence
        bodys_sequence = []
        bodys_probability = []
        action_name = []
        for body in skeletal_coordinates:
            body_sequence = []
            body_probability = 0.0
            #luu ten cac bo phan va vi tri cac bo phan cho tung body pose
            xys = {}
            #inference_box la box sau khi luu
        
            box_x, box_y, box_w, box_h = inference_box
            scale_x, scale_y = src_size[0] / box_h, src_size[1] / box_w
            
            #print("scale_x, scale_y",scale_x, scale_y)
            
            # For each joint put it in vetcor list
            
            for label, keypoint in body.keypoints.items():
                
                kp_x = int((keypoint.point[0] - box_x) * scale_x)
                kp_y = int((keypoint.point[1] - box_y) * scale_y)
                
                body_sequence.append(kp_x)
                
                body_sequence.append(kp_y)
                
                xys[label] = (kp_x, kp_y)
                
                # Sum joints probability to find body probability
                body_probability += keypoint.score

            abc = np.array([body_sequence])
            preds = xgb_model_loaded.predict(abc)  
            print("Hanh dong",str(preds))
            #======================================================= 
                
            body_probability = body_probability/len(xys)
            
            # Add body sequence to list
            bodys_sequence.append(body_sequence)
            bodys_probability.append(body_probability)
            action_name.append(preds)
            '''
            for a, b in self.EDGES:
                if a not in xys or b not in xys: continue
                ax, ay = xys[a]
                bx, by = xys[b]
                cv2.line(self.image,(ax, ay), (bx, by),(0,0,255))
            # Assign coordiates sequence
            '''
        return [bodys_sequence, bodys_probability,action_name]

    '''
        Evaluate ellipses shadow, for each body
    '''

    def evaluate_ellipses(self, skeletal_coordinates, draw_ellipse_requirements, ellipse_boxes, ellipse_pool,show_dis_from_cam):
        for skeletal_coordinate in skeletal_coordinates[0]:
            
            # Evaluate skeleton bounding box
            left, right, top, bottom = self.compute_simple_bounding_box(
                np.array(skeletal_coordinate))
            print("skeletal_coordinate",skeletal_coordinate)
            print("lrtb", left, right, top, bottom)
            bb_center = np.array(
                [(left + right) / 2, (top + bottom) / 2], np.int32)
            
            
            calculated_height = self.evaluate_height(skeletal_coordinate)
            print("calculated_height", calculated_height)
            
            

            # computing how the height of the circle varies in perspective
            pts = np.array(
                [[bb_center[0], top], [bb_center[0], bottom]], np.float32)
            pts1 = pts.reshape(-1, 1, 2).astype(np.float32)  # (n, 1, 2)
            dst1 = cv2.perspectiveTransform(pts1, self.homography_matrix)
            # height of the ellipse in perspective
            width = int(dst1[1, 0][1] - dst1[0, 0][1])
            # Bounding box surrending the ellipses, useful to compute whether there is any overlap between two ellipses
            ellipse_bbx = [bb_center[0] - calculated_height,
                           bb_center[0] + calculated_height, bottom - width, bottom + width]
            # Add boundig box to ellipse list
            ellipse_boxes.append(ellipse_bbx)

            ellipse = [int(bb_center[0]), int(bottom),
                       int(calculated_height), int(width)]

            mask_copy = self.mask1.copy()
            
            ellipse_pool.append(cv2.ellipse(mask_copy, (int(bb_center[0]/self.overlap_precision), int(bottom/self.overlap_precision)), (int(
                calculated_height/self.overlap_precision), int(width/self.overlap_precision)), 0, 0, 360, (255, 255, 255), -1))
            
            distance_from_camera_to_person = self.distance_to_camera(150.0,6.6,calculated_height*0.0464) # 1 pixel = 0.026458333 cm

            distance_f_camera = int(0 if distance_from_camera_to_person is None else distance_from_camera_to_person)
            #print("bb_center",bb_center[0],bb_center[1])
            distance_from_cam = [distance_f_camera,bb_center[0],bb_center[1]]
            show_dis_from_cam.append(distance_from_cam)
            
            draw_ellipse_requirements.append(ellipse)


    def distance_to_camera(self,knownWidth, focalLength ,perWidth):
        if(perWidth != 0):
	    # compute and return the distance from the maker to the camera 
            return (knownWidth * focalLength) / perWidth

    
    '''
        Analyze image and evaluate distances
    '''

    def distances_evaluate(self, image, background,xgb_model_loaded):
        ellipse_boxes = []

        draw_ellipse_requirements = []

        ellipse_pool = []
        
        show_dis_from_cam = []

        cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_im = Image.fromarray(cv2_im_rgb)
        
        h_cap, w_cap, _ = image.shape
        #==============FPS===================================================================
        n = 0
        sum_process_time = 0
        sum_inference_time = 0
        ctr = 0
        start_time = time.monotonic()
        #============================================================================================================================
        src_size = (h_cap,w_cap)
        print("src_size",src_size)
        
        #poses, inference_time = self.engine.DetectPosesInImage(np.uint8(pil_im.resize((641, 481), Image.NEAREST)))
        poses, inference_time = self.engine.DetectPosesInImage(pil_im)
        input_shape = self.engine.get_input_tensor_shape()

        
        inference_size = (input_shape[2], input_shape[1])
        inference_box = (0,0,input_shape[2],input_shape[1])
        
        print("inference_size",inference_size)
        
        skeletal_coordinates = poses

        # Trace on background
        if self.background_masked:
            image = background
        
        background = cv2.resize(background, (641, 481))
        

        #self.dt_vector['ts'] = int(round(time.time() * 1000))
        #self.dt_vector['bodys'] = []

        if type(skeletal_coordinates) is list:
            # Remove probability from joints and get a joint position list
            skeletal_coordinates = self.create_joint_array(
                skeletal_coordinates,inference_box,src_size,xgb_model_loaded)

            # Initialize overlapped buffer
            is_skeletal_overlapped = np.zeros(
                np.shape(skeletal_coordinates[0])[0])

            # Evaluate ellipses for each body detected by openpose
            self.evaluate_ellipses(skeletal_coordinates,
                                   draw_ellipse_requirements, ellipse_boxes, ellipse_pool, show_dis_from_cam)

            # Evaluate overlapping
            self.evaluate_overlapping(
                ellipse_boxes, is_skeletal_overlapped, ellipse_pool)

             #===============================================================================================================================
            end_time = time.monotonic()
            n += 1
            sum_process_time += 1000 * (end_time - start_time)
            sum_inference_time += inference_time
            avg_inference_time = sum_inference_time / n
            ##text_line = 'PoseNet: %.1fms (%.2f fps) Number of poses %d' % (avg_inference_time, 1000 / avg_inference_time, len(poses))
            #text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f' % (avg_inference_time, 1000 / avg_inference_time, next(fps_counter)) 
            #cv2_im = cv2.rectangle(cv2_im, (0), 0), (x1, y1), (0, 255, 0), 2)
            #cv2.putText(image,text_line, (100,100), 0, 2, 20)
            ##cv2.putText(image, text_line , (0,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0,0), 2)
            ##print(text_line)
            #=================================================================================================================================
       
            # Trace results over output image
            self.trace(image, skeletal_coordinates,
                        draw_ellipse_requirements, is_skeletal_overlapped,show_dis_from_cam)
        
        return image
    
    
    def send_image(self, queue_list, image, ts):

        encoded_image = self.jpeg.encode(image, quality=80)
        # Put image into queue for each server thread
        for q in queue_list:
            try:
                block = (ts, encoded_image)
                q.put(block, True, 0.02)
            except queue.Full:
                pass

    def analyze_video(self):
        first_frame = True
        f = open("demofile2.txt", "w")
        framcount = 0
        cap1 = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(1)
        #cap3 = cv2.VideoCapture(2)
        #cap4 = cv2.VideoCapture(3)
        #============================================================================================================================
        #################load models human action recognition #########################
    
        time.sleep(0.1)
        filename = 'models/yoga_poses.sav'
        
        xgb_model_loaded = pickle.load(open(filename, "rb"))
        print("xgb_model_loaded",xgb_model_loaded)
        ###########################################################################################
        #out = cv2.VideoWriter(str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))+'.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
        f = open("demofile2.txt", "w")
        '''
        while self.cap.isOpened():
            # Capture from image/video
            ret, image = self.cap.read()
            image = cv2.resize(image,(640,480))
            
            # Check image
            if image is None or not ret:
                os._exit(0)

            # create a mask from image
            if first_frame:
                self.mask = np.zeros(
                    (int(image.shape[0]/self.overlap_precision), 
                    int(image.shape[1]/self.overlap_precision), image.shape[2]), dtype=np.uint8)
                first_frame = False
        
            # Get openpose output
            if self.background_masked:
                background = self.background_image.copy()
            else:
                background = image
            
            image = self.distances_evaluate(image, background,xgb_model_loaded)
            
            
            #write image 
            #self.out.write(image)
            
            cv2.imshow('Social Distance', image)
            
            
            
            
            cv2.waitKey(1)
        '''        
        while (cap1.isOpened() and cap2.isOpened()):
            # Capture from image/video
            ret1, image1 = cap1.read()
            ret2, image2 = cap2.read()
            #ret3, image3 = cap3.read()
            #ret4, image4 = cap4.read()
            image1 = cv2.resize(image1,(640,480))
            image2 = cv2.resize(image2,(640,480))
            #image3 = cv2.resize(image3,(640,480))
            #image4 = cv2.resize(image4,(640,480))
            
            # Check image
            
            if image1 is None or not ret1:
                os._exit(0)

            # create a mask from image
            if first_frame:
                self.mask1 = np.zeros(
                    (int(image1.shape[0]/self.overlap_precision), 
                    int(image1.shape[1]/self.overlap_precision), image1.shape[2]), dtype=np.uint8)
                self.mask2 = np.zeros(
                    (int(image2.shape[0]/self.overlap_precision), 
                    int(image2.shape[1]/self.overlap_precision), image2.shape[2]), dtype=np.uint8)
                first_frame = False
        
            # Get openpose output
            if self.background_masked:
                background1 = self.background_image.copy()
                background2 = self.background_image.copy()
                #background3 = self.background_image.copy()
                #background4 = self.background_image.copy()
            else:
                background1 = image1
                background2 = image2
                #background3 = image3
                #background4 = image4
            
            image1 = self.distances_evaluate(image1, background1,xgb_model_loaded)
            image2 = self.distances_evaluate(image2, background2,xgb_model_loaded)
            #image3 = self.distances_evaluate(image3, background3,xgb_model_loaded)
            #image4 = self.distances_evaluate(image4, background4,xgb_model_loaded)
            
            
            #write image 
            #self.out.write(image)
            
            cv2.imshow('Social Distance', image1)
            cv2.imshow('ABC Distance', image2)
            #cv2.imshow('Came3 Distance', image3)
            #cv2.imshow('Cam4 Distance', image4)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
            cv2.waitKey(1)
            
                
    '''
        Analyze image
    '''

    def analyze_image(self):
        
        self.image = cv2.resize(self.image,(640,480))
        # Get openpose output
        if self.background_masked:
            background = self.background_image.copy()
        else:
            background = self.image
        
        
        
        # Create mask from image
        self.mask = np.zeros(
            (int(self.image.shape[0]/self.overlap_precision), 
            int(self.image.shape[1]/self.overlap_precision), self.image.shape[2]), dtype=np.uint8)

        self.image = self.distances_evaluate(self.image, background)

        # Write image
        cv2.imwrite(self.image_out, self.image)
        
        
        
        # Show image and wait some time
        cv2.imshow('Social Distance', self.image)
        cv2.waitKey(1000)
            

    '''
        Analyze image/video
    '''

    def analyze(self):
        if self.use_image:
            self.analyze_image()
        
        if self.use_video:
            self.analyze_video()
 
'''
    Main Entry
'''
if __name__ == "__main__":
    
    default_model_dir = 'models'
    default_model = 'mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    default_labels = 'hand_label.txt'
    # Argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    
    parser.add_argument("--video", default="disabled",
                        help="select video mode, if defined")

    parser.add_argument("--image", default="disabled",
                        help="select image mode, if defined")

    parser.add_argument("--masked", default="disabled",
                        help="mask to blur visual appearance of people")

    parser.add_argument("--image_in", default="samples/298.jpg",
                        help="Process an image. Read all standard image formats")

    parser.add_argument("--image_out", default="./output_image.jpg",
                        help="Image output")

    parser.add_argument("--background_in", default="samples/286.jpg",
                        help="Process an image, read all standard formats (jpg, png, bmp, etc.).")

    parser.add_argument("--stream_in", default="stream_in.mp4",
                        help="Process an image ora a video stream. Read all standard formats and connect to live stream")

    parser.add_argument("--stream_out", default="./output_stream.avi",
                        help="Image/video output")

    parser.add_argument("--net_size", default="640x480",
                        help="Openpose network size")

    parser.add_argument("--horizontal_ratio", default="0.7",
                        help="Ratio between the closest horizotal line of the scene to the furthest visible. It must be a float value in (0,1)")

    parser.add_argument("--vertical_ratio", default="0.7",
                        help="Ratio between the height of the trapezoid wrt the rectangular bird’s view scene (image height). It must be a float value in (0,1)")

    parser.add_argument("--calibration", default="1.0",
                        help="calibrate each point of view with this value")

    parser.add_argument("--body_threshold", default="0.2",
                        help="remove too low confidential body")

    parser.add_argument("--show_confidence", default="disabled",
                        help="show confidence value")

    parser.add_argument("--overlap_precision", default="2",
                        help="lower better, from 1 to 16, create ellipses image sub-size mask")

    parser.add_argument("--encoding_codec", default="XVID",
                        help="change output video encoding mode")
    
    # Parsing arguments
    args = parser.parse_known_args()
    
    # Create social_distance object
    social_distance = SocialDistancing(args)
    print("Start working")
    # Do hard work
    social_distance.analyze()
