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

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
from __future__ import division, print_function, absolute_import
import argparse
import cv2
import os
from timeit import time
import warnings
import numpy as np
from PIL import Image

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video



from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


def main():
    default_model_dir = '../models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_model_extract_em = 'resnet_edgetpu.tflite'#'efficientnet-edgetpu-M_quant_embedding_extractor_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--model_extract_em', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model_extract_em))                    
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()
    #Definition of the parameters for tracking 
    max_cosine_distance = 12
    nn_budget = None
    nms_max_overlap = 1.0
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    tracking = True

    #
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    interpreter_extactor = make_interpreter(args.model_extract_em, device=':0')
    interpreter_extactor.allocate_tensors()
    
    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv2_im,boxes,scores,classes,features = append_objs_to_img(cv2_im,interpreter_extactor, inference_size, objs, labels)
        print("boxes",len(boxes))
        print("classes",len(classes))
        print("features",len(features))
        detections = [Detection(bbox, confidence, clss, feature[0]) for bbox, confidence, clss, feature in
                      zip(boxes, scores, classes, features)]
        for det in detections:
            bbox = det.to_tlbr()
            print("bbox",bbox)
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

        #indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        #detections = [detections[i] for i in indices]
        print("detections",detections)
        
        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

        #cv2.imshow('', frame)
        height, width, channels = cv2_im.shape
        cv2_im = cv2.resize(cv2_im, (width *3,height*3))
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def extract_embeddings(img, interpreter):
    """Uses model to process images as embeddings.
    Reads image, resizes and feeds to model to get feature embeddings. Original
    image is discarded to keep maximum memory consumption low.
    Args:
        image_paths: ndarray, represents a list of image paths.
        interpreter: TFLite interpreter, wraps embedding extractor model.
    Returns:
        ndarray of length image_paths.shape[0] of embeddings.
    """
    input_size = common.input_size(interpreter)
    feature_dim = classify.num_classes(interpreter)

    embeddings = np.empty((1, feature_dim), dtype=np.float32)
    print("embeddings",embeddings,feature_dim)
    #embeddings[0,:] = classify.get_scores(interpreter)
    #print("embeddings1",embeddings)
    #common.set_input(interpreter, img.resize(input_size, Image.NEAREST))
    #interpreter.invoke()
    #embeddings[count,:] = classify.get_scores(interpreter)
    
    return embeddings

def append_objs_to_img(cv2_im,interpreter_extactor, inference_size, objs, labels):
    features = [] #!
    image_patches = [] #1
    boxes = []
    scores = []
    classes = []
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs: #0.1
        bbox = obj.bbox.scale(scale_x, scale_y)
        
        #end1
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        #print("x0, y0,x1, y1",(x0/scale_x)/inference_size[0], (y0/scale_y)/inference_size[1],(x1/scale_x)/inference_size[0], (y1/scale_y)/inference_size[1])
        #cv2.imshow('frame_cut', cv2_im[y0:y1, x0:x1])
        feature = extract_embeddings(cv2_im[y0:y1, x0:x1], interpreter_extactor)
        #print("feature",feature)
        features.append(feature)
        
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        boxes.append(bbox)
        scores.append(obj.score)
        classes.append(labels.get(obj.id, obj.id))
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im,np.array(boxes),np.array(scores),np.array(classes),np.array(features)

if __name__ == '__main__':
    main()
