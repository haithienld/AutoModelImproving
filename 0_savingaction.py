# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
##
import xgboost as xgb
import argparse
import collections
from functools import partial
import re
import time

import numpy as np
from PIL import Image
import svgwrite
import gstreamer
#####
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

def __init__(self, args):
        # Ratio params
        self.engine = PoseEngine(args[0].model)
        
        horizontal_ratio = float(args[0].horizontal_ratio)
        vertical_ratio = float(args[0].vertical_ratio)


def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))


def draw_pose(cv2_im,f,framcount,xgb_model_loaded, pose, action, src_size, inference_box, color='yellow', threshold=0.2):
    #box_x, box_y, box_w, box_h = inference_box

    box_x = 0
    box_y = 0  
    box_w = 641
    box_h = 480
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    dem = 0
    fullkeypoint=[]
    for label, keypoint in pose.keypoints.items():
        #if keypoint.score < threshold: continue
        if keypoint.score < threshold:
            kp_x1 = 0
            kp_y1 = 0
            fullkeypoint.append(kp_x1)
            fullkeypoint.append(kp_y1)
        else:
        # Offset and scale to source coordinate space.
            kp_x = int((keypoint.point[0] - box_x) * scale_x)
            kp_y = int((keypoint.point[1] - box_y) * scale_y)
            fullkeypoint.append(kp_x)
            fullkeypoint.append(kp_y)
        #fullkeypoint.append(kp_y)
            xys[label] = (kp_x, kp_y)
            cv2.circle(cv2_im,(int(kp_x),int(kp_y)),5,(0,255,255),-1)
        #dwg.add(dwg.circle(center=(int(kp_x), int(kp_y)), r=5,
        #                   fill='cyan', fill_opacity=keypoint.score, stroke=color))
        dem +=1
        #if (dem >=16):
    #Action recognition
    abc = np.array([fullkeypoint])
    preds = xgb_model_loaded.predict(abc)
    print("Hanh dong",str(preds))

    cv2.putText(cv2_im,str(preds),
                (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    
    
    f.write(str(framcount) + "," + str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))+","+str(fullkeypoint)+","+str(action))
    f.write('\n')
    encoded, buffer = cv2.imencode('.jpg', cv2_im)
    jpg_as_text = base64.b64encode(buffer)


   
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        cv2.line(cv2_im,(ax, ay), (bx, by),(0,255,255))
        #dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))
   
        
def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


#===========================================================
def main():
    #default_model_dir = '../all_models'
    #default_model = 'posenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    #parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    parser.add_argument('--action', type=str, help='name of action',default='standing', required=False)
    args = parser.parse_args()



    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])

    cap = cv2.VideoCapture(args.camera_idx)
    #cap = cv2.VideoCapture('media/camerasongsong2.mp4')
    #################load models human action recognition #########################
    
    time.sleep(0.1)
    filename = 'models/yoga_poses.sav'
    xgb_model_loaded = pickle.load(open(filename, "rb"))
    print("xgb_model_loaded",xgb_model_loaded)
    ###########################################################################################
    #out = cv2.VideoWriter(str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))+'.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    f = open("action_label.txt", "w")
    framcount = 0
    while cap.isOpened():
    #while cv2.waitKey(1)<0:
        ret, frame = cap.read()
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame,(640,480))
        
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_im_rgb)
        #declare new window for show pose in 2d plane========================
        h_cap, w_cap, _ = cv2_im.shape
        cv2_sodidi = np.zeros((h_cap,w_cap,3), np.uint8)
        #======================================pose processing=================================
        poses, inference_time = engine.DetectPosesInImage(pil_image)
        #poses, _ = engine.DetectPosesInImage(np.uint8(pil_im.resize((640, 480), Image.NEAREST)))
        #print('Posese is',poses)
        input_shape = engine.get_input_tensor_shape()
        inference_size = (input_shape[2], input_shape[1])
        print("shape",len(poses))
        if(len(poses) > 0):
            draw_pose(cv2_im,f,framcount,xgb_model_loaded, poses[0],args.action,src_size, 0)
        #for pose in poses:
        #    draw_pose(cv2_im,f,framcount, pose,src_size, 0)

        cv2.imshow('frame', cv2_im)    
        #out.write(cv2_im)
        framcount += 1
        #==============================================================
        #encoded, buffer = cv2.imencode('.jpg', cv2_im)
        #jpg_as_text = base64.b64encode(buffer)
        #footage_socket.send(jpg_as_text)

        #=====================================================================
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    f.close()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
