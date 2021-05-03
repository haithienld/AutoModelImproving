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

"""A demo that runs hand tracking and object detection on camera frames using OpenCV. 2 EDGETPU
"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
import math
from PIL import Image
import re
from edgetpu.detection.engine import DetectionEngine

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import base64
import itertools
import cv2
import zmq

import time
import svgwrite
import gstreamer
from pose_engine import PoseEngine
import tflite_runtime.interpreter as tflite

#===========streamming======================
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://147.46.123.186:4664') 
#===========================================
Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

#==============================
EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

HEADCHECK = ('nose', 'left eye','right eye' ,'left ear', 'right ear')
SHOULDERCHECK = ('left shoulder', 'right shoulder') 
HIPCHECK = ('left hip','right hip')
KNEECHECK = ('left knee','right knee')
ANKLECHECK = ('left ankle','right ankle')

def shadow_text(cv2_im, x, y, text, font_size=16):
    cv2_im = cv2.putText(cv2_im, text, (x + 1, y + 1),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    #dwg.add(dwg.text(text, insert=, fill='black',
    #                 font_size=font_size, style='font-family:sans-serif'))
    #dwg.add(dwg.text(text, insert=(x, y), fill='white',
    #                 font_size=font_size, style='font-family:sans-serif'))

def draw_pose(cv2_im, cv2_sodidi, pose, numobject, src_size, color='yellow', threshold=0.2):
    box_x = 0
    box_y = 0  
    box_w = 641
    box_h = 480
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    #==bien dung de tinh khoang cach giua cac bo phan trong co the ============
    pts_sodidi = []
    headarea={}
    shoulderarea={}
    elbow={}
    lengbackbone=60
    lengleg= 86
    lengface = 30
    #=======================================================
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
        kp_y = int((keypoint.yx[0] - box_y) * scale_y)
        kp_x = int((keypoint.yx[1] - box_x) * scale_x)
        cv2_im = cv2.putText(cv2_im, str(numobject),(kp_x + 1, kp_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        xys[label] = (numobject,kp_x, kp_y)
        cv2.circle(cv2_im,(int(kp_x),int(kp_y)),5,(0,255,255),-1)

    #draw pose in 2d plane========================
    checkankle, checkknee,checkhip,checkshoulder,checkhead = False,False, False,False,False
    knee,hip,shoulder,head = {},{},{},{}
    pts_in = np.array([[1.0, 1.0]], dtype='float32')
    for a in ANKLECHECK:
        if a in xys:
            _,x1,y1 = xys[a]
            pts_in = np.array([[x1, y1]], dtype='float32')
        else:
            for b in KNEECHECK:
                if b in xys:
                    checkknee = True
                    knee = xys[b]
            for c in HIPCHECK:
                if c in xys:
                    checkhip = True
                    hip = xys[c]
            for d in SHOULDERCHECK:
                if d in xys:
                    checkshoulder = True
                    shoulder = xys[d]
            for e in HEADCHECK:
                if e in xys:
                    checkhead = True
                    head = xys[e]
            if checkknee == True and checkhip == True:
                _,x1,y1 = knee
                _,x2,y2 = hip
                leeeng =  check_distance(x1,y1,x2,y2)
                pts_in = np.array([[x1, y1 + leeeng/2]], dtype='float32')
                break
            if checkhip == True and checkshoulder == True:
                _,x1,y1 = shoulder
                _,x2,y2 = hip
                leeeng =  check_distance(x1,y1,x2,y2)
                pts_in = np.array([[x2, y2 + lengleg*leeeng/lengbackbone]], dtype='float32')
                break
            if checkhead == True and checkshoulder == True:
                _,x1,y1 = shoulder
                _,x2,y2 = head
                leeeng =  check_distance(x1,y1,x2,y2)
                pts_in = np.array([[x1, y1 + (lengleg+lengbackbone)*leeeng/lengface]], dtype='float32')
                break

    pts_in = np.array([pts_in])
    pts_out = mapcamto2dplane(pts_in)
    #print(len(pts_out))
    #print(pts_out[0][0,0])
    #print(pts_out[0][0,1])
    
    pts_sodidi = np.array([numobject,pts_out[0][0,0],pts_out[0][0,1]])
    #cv2_sodidi = cv2.circle(cv2_sodidi,(int(pts_out[0][0,0]),int(pts_out[0][0,1])),5,(0,255,255),-1)
        #=============================================
    return pts_sodidi, xys


    '''
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        anum,ax, ay = xys[a]
        bnum,bx, by = xys[b]
        print(numobject,a,xys[a],b,xys[b])
        cv2.line(cv2_im,(ax, ay), (bx, by),(0,255,255))
    '''

def mapcamto2dplane(pts_in):
    # provide points from image 1
    pts_src = np.array([[7, 476], [6, 185], [635, 138],[638, 477], [1, 191]])
    # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
    pts_dst = np.array([[7, 476], [9, 8], [636, 8],[638, 477], [1, 191]])#np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])

    # calculate matrix H
    h, status = cv2.findHomography(pts_src, pts_dst)

    # provide a point you wish to map from image 1 to image 2
    #pts_in = np.array([[154, 174]], dtype='float32')
    #pts_in = np.array([pts_in])

    # finally, get the mapping
    pointsOut = cv2.perspectiveTransform(pts_in, h)
    pointsOut = np.array([pointsOut])
    point_out = [b for a in pointsOut for b in a]
    return point_out

def check_distance(x1,y1,x2,y2):
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


#==============================
def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

#===GET POSITION TO MAPPING BETWEEN 2 WINDOW================
posList=[]
def onMouse(event, x,y, flags, param):
    global posList
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x,y))

#===========================================================
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
            (0, 0, 153), (0, 0, 153),
            (0, 153, 153), (0, 153, 153),
            (0, 153, 153)
            ]

connections = [(0, 16), (0, 15), (16, 18), (15, 17),
               (0, 1), (1, 2), (2, 3), (3, 4),
               (1, 5), (5, 6), (6, 7), (1, 8),
               (8, 9), (9, 10), (10, 11),
               (8, 12), (12, 13), (13, 14),
               (11, 24), (11, 22), (22, 23),
               (14, 21), (14, 19), (19, 20)]

'''
Draw Skelethon
'''
def draw_skeleton(frame, keypoints, colour):

    for keypoint_id1, keypoint_id2 in connections:
        x1, y1 = keypoints[keypoint_id1]
        x2, y2 = keypoints[keypoint_id2]

        if 0 in (x1, y1, x2, y2):
            continue

        pt1 = int(round(x1)), int(round(y1))
        pt2 = int(round(x2)), int(round(y2))

        cv2.circle(frame, center=pt1, radius=4,
                   color=nd_color[keypoint_id2], thickness=-1)
        cv2.line(frame, pt1=pt1, pt2=pt2,
                 color=nd_color[keypoint_id2], thickness=2)

'''
    Compute skelethon bounding box
'''

def compute_simple_bounding_box(skeleton):
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

def compute_homography(H_ratio, V_ratio, im_size):
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

def compute_overlap(rect_1, rect_2):
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

def trace(image, skeletal_coordinates, draw_ellipse_requirements, is_skeletal_overlapped):
    bodys = []

    # Trace ellipses and body on target image
    i = 0

    for skeletal_coordinate in skeletal_coordinates[0]:
        if float(skeletal_coordinates[1][i]) < 0.2: #body_threshold=0.2
            continue

        # Trace ellipse
        cv2.ellipse(image,
                    (int(draw_ellipse_requirements[i][0]), int(
                        draw_ellipse_requirements[i][1])),
                    (int(draw_ellipse_requirements[i][2]), int(
                        draw_ellipse_requirements[i][3])), 0, 0, 360,
                    colors[int(is_skeletal_overlapped[i])], 3)

        # Trace skelethon
        skeletal_coordinate = np.array(skeletal_coordinate)
        draw_skeleton(
            image, skeletal_coordinate.reshape(-1, 2), (255, 0, 0))

        if int(skeletal_coordinate[2]) != 0 and int(skeletal_coordinate[3]) != 0 and show_confidence:
            cv2.putText(image, "{0:.2f}".format(skeletal_coordinates[1][i]),
                        (int(skeletal_coordinate[2]), int(skeletal_coordinate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Append json body data, joints coordinates, ground ellipses
        bodys.append([[round(x) for x in skeletal_coordinate],
                      draw_ellipse_requirements[i], int(is_skeletal_overlapped[i])])

        i += 1

    dt_vector["bodys"] = bodys

'''
    Evaluate skelethon height
'''

def evaluate_height(skeletal_coordinate):
    # Calculate skeleton height
    calculated_height = 0
    pointer = -1

    # Left leg
    joint_set = [12, 13, 14]

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
        joint_set = [9, 10, 11]
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
    return calculated_height * 1.0 #calibrate = 1.0

'''
Evaluate overlapping
'''

def compute_overlap(rect_1, rect_2):
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

def trace(image, skeletal_coordinates, draw_ellipse_requirements, is_skeletal_overlapped):
    bodys = []

    # Trace ellipses and body on target image
    i = 0

    for skeletal_coordinate in skeletal_coordinates[0]:
        if float(skeletal_coordinates[1][i]) < 0.2:
            continue

        # Trace ellipse
        cv2.ellipse(image,
                    (int(draw_ellipse_requirements[i][0]), int(draw_ellipse_requirements[i][1])),
                    (int(draw_ellipse_requirements[i][2]), int(draw_ellipse_requirements[i][3])), 0, 0, 360,
                    colors[int(is_skeletal_overlapped[i])], 3)

        # Trace skelethon
        skeletal_coordinate = np.array(skeletal_coordinate)
        draw_skeleton(
            image, skeletal_coordinate.reshape(-1, 2), (255, 0, 0))

        if int(skeletal_coordinate[2]) != 0 and int(skeletal_coordinate[3]) != 0 : #and self.show_confidence
            cv2.putText(image, "{0:.2f}".format(skeletal_coordinates[1][i]),
                        (int(skeletal_coordinate[2]), int(skeletal_coordinate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Append json body data, joints coordinates, ground ellipses
        bodys.append([[round(x) for x in skeletal_coordinate],draw_ellipse_requirements[i], int(is_skeletal_overlapped[i])])

        i += 1

    return bodys

'''
Evaluate skelethon height
'''

def evaluate_height(self, skeletal_coordinate):
    calibrate = 1.0
    # Calculate skeleton height
    calculated_height = 0
    pointer = -1

    # Left leg
    joint_set = [12, 13, 14]

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
        joint_set = [9, 10, 11]
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
    return calculated_height * calibrate

'''
Evaluate overlapping
'''
def evaluate_overlapping(ellipse_boxes, is_skeletal_overlapped, ellipse_pool):
    # checks for overlaps between people's ellipses, to determine risky or not
    for ind1, ind2 in itertools.combinations(list(range(0, len(ellipse_pool))), 2):

        is_overlap = cv2.bitwise_and(
            ellipse_pool[ind1], ellipse_pool[ind2])

        if is_overlap.any() and (not is_skeletal_overlapped[ind1] or not is_skeletal_overlapped[ind2]):
            is_skeletal_overlapped[ind1] = 1
            is_skeletal_overlapped[ind2] = 1
    return is_skeletal_overlapped
'''
    Create Joint Array
'''
def create_joint_array(skeletal_coordinates):
        # Get joints sequence
        bodys_sequence = []
        bodys_probability = []
        for body in skeletal_coordinates:
            body_sequence = []
            body_probability = 0.0
            # For each joint put it in vetcor list
            for label, keypoint in body.keypoints.items():
                print(keypoint)
                if keypoint.score < 0.2: continue
                body_sequence.append(keypoint.yx[0])
                body_sequence.append(keypoint.yx[1])
                
                # Sum joints probability to find body probability
                body_probability += keypoint.score
    
           

            # Add body sequence to list
            bodys_sequence.append(body_sequence)
            bodys_probability.append(body_probability)
        # Assign coordiates sequence
        return [bodys_sequence, bodys_probability]


def evaluate_ellipses(skeletal_coordinates, draw_ellipse_requirements, ellipse_boxes, ellipse_pool,mask,homography_matrix):
    overlap_precision = 2.0
    for skeletal_coordinate in skeletal_coordinates:
        # Evaluate skeleton bounding box
        left, right, top, bottom = self.compute_simple_bounding_box(
            np.array(skeletal_coordinate))

        bb_center = np.array(
            [(left + right) / 2, (top + bottom) / 2], np.int32)

        calculated_height = evaluate_height(skeletal_coordinate)

        # computing how the height of the circle varies in perspective
        pts = np.array(
            [[bb_center[0], top], [bb_center[0], bottom]], np.float32)

        pts1 = pts.reshape(-1, 1, 2).astype(np.float32)  # (n, 1, 2)

        dst1 = cv2.perspectiveTransform(pts1, homography_matrix)

        # height of the ellipse in perspective
        width = int(dst1[1, 0][1] - dst1[0, 0][1])

        # Bounding box surrending the ellipses, useful to compute whether there is any overlap between two ellipses
        ellipse_bbx = [bb_center[0]-calculated_height,
                       bb_center[0]+calculated_height, bottom-width, bottom+width]

        # Add boundig box to ellipse list
        ellipse_boxes.append(ellipse_bbx)

        ellipse = [int(bb_center[0]), int(bottom),
                   int(calculated_height), int(width)]

        mask_copy = mask.copy()

        ellipse_pool.append(cv2.ellipse(mask_copy, (int(bb_center[0]/overlap_precision), int(bottom/overlap_precision)), (int(
            calculated_height/overlap_precision), int(width/overlap_precision)), 0, 0, 360, (255, 255, 255), -1))

        draw_ellipse_requirements.append(ellipse)
    return draw_ellipse_requirements
#=============social distancing===========================



def main():
    default_model_dir = '../all_models'
    default_model = 'posenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    default_labels = 'hand_label.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    
    args = parser.parse_args()
    W = 640
    H = 480
    src_size = (W, H)
    print('Loading Pose model {}'.format(args.model))
    engine = PoseEngine(args.model)
    cap = cv2.VideoCapture(args.camera_idx)
    #==============social distancing=========================
    dt_vector = {}
    first_frame = True
    overlap_precision = 2 #"lower better, from 1 to 16, create ellipses image sub-size mask"
    
    #=======================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame:
                mask = np.zeros(
                    (int(frame.shape[0]/overlap_precision), 
                    int(frame.shape[1]/overlap_precision), frame.shape[2]), dtype=np.uint8)
                first_frame = False
 
        background = frame

        ellipse_boxes = []

        draw_ellipse_requirements = []

        ellipse_pool = []
        
        cv2_im = frame
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        im_size =  (frame.shape[1], frame.shape[0])
        horizontal_ratio = 0.7
        vertical_ratio = 0.7
        homography_matrix = compute_homography(horizontal_ratio, vertical_ratio, im_size)
        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_im.resize((641, 481), Image.NEAREST)))       
        
        
        skeletal_coordinates = poses
        
        dt_vector['ts'] = int(round(time.time() * 1000))
        dt_vector['bodys'] = []
        
        if type(skeletal_coordinates) is list:
            # Remove probability from joints and get a joint position list
            skeletal_coordinates = create_joint_array(
                skeletal_coordinates)
     
            print(skeletal_coordinates[0])


            is_skeletal_overlapped = np.zeros(
                    np.shape(skeletal_coordinates[0])[0])


            # Evaluate ellipses for each body detected by openpose
            draw_ellipse_requirements = evaluate_ellipses(skeletal_coordinates[0], draw_ellipse_requirements, 
                                                          ellipse_boxes, ellipse_pool,mask,homography_matrix)

            # Evaluate overlapping
            is_skeletal_overlapped = evaluate_overlapping(ellipse_boxes, is_skeletal_overlapped, ellipse_pool)
            # Trace results over output image
            bodys = trace(cv2_im, skeletal_coordinates,draw_ellipse_requirements, is_skeletal_overlapped)
            dt_vector["bodys"] = bodys
        
        #===============================
        
        cv2_sodidi = cv2_im
        cv2.imshow('frame', cv2_im)
        cv2.imshow('1', cv2_sodidi)
        
        #===========print mouse pos=====================
        cv2.setMouseCallback('frame',onMouse)
        posNp=np.array(posList)
        print(posNp)
        #============streamming to server==============
        img = np.hstack((cv2_im, cv2_sodidi))
        #thay frame = img
        encoded, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)
        #=============================================
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()