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
import argparse
import cv2
import os
import base64
import json
import time 
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from datetime import datetime

#str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
def main():
    default_model_dir = '../models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        original = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)
        
        if(frame_count > 0):
            s,m = compare_images(original,lastframe)
            print("frame_count, s,m",frame_count, s,m)
            if(m < 0.6):
                break
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels,frame_count)
        frame_count += 1
        height, width, channels = cv2_im.shape
        cv2_im = cv2.resize(cv2_im, ( width*2, height*2))
        cv2.imshow('frame', cv2_im)
        lastframe = original 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    return m,s


def save(filename,
        __version__,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        #if imageData is not None:
            #imageData = base64.b64encode(imageData).decode("utf-8")
            #imageHeight, imageWidth = _check_image_height_and_width(
            #    imageData, imageHeight, imageWidth
            #)
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
            
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open("image/" + filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                #filename = filename
        except Exception as e:
            raise LabelFileError(e)
def append_objs_to_img(cv2_im, inference_size, objs, labels,frame_count):
    
    shapes =[]
    #cv2.imwrite("image/frame%d.jpg" % frame_count, cv2_im)
    #=====================contour===================
    imgContour = cv2_im.copy()
    #Convert to grayscale image
    imgGray = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)
    #blurred image 
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    #edge detection 
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        color1 = (list(np.random.choice(range(256), size=3)))  
        color =[int(color1[0]), int(color1[1]), int(color1[2])]  

        #Positioning area
        area = cv2.contourArea(cnt)
        #print(area)
        #Retrieve minimum area
        if area > 100 and area < 500:
            # Draw outline area
            cv2.drawContours(imgContour, cnt, -1, color, 3) # color (255, 0, 0)
            #Calculate curve length
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            #Calculate corner points
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            #Create object corners
            objCor = len(approx)
            #Get bounding box boundary
            x, y, w, h = cv2.boundingRect(approx)
            # Classifying objects
            if objCor == 3 : objectType = "Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio< 1.05: objectType = "Square"
                else: objectType = "Rectangle"
            elif objCor > 4: objectType = "Circles"
            else: objectType="None"
            #Draw a rectangular bounding box
            cv2_im = cv2.rectangle(cv2_im, (x, y), (x+w, y+h), color, 3) # color (0, 255, 0)
            cv2_im = cv2.putText(cv2_im, objectType, (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
            dict_shape_contour = {"label":objectType,"points":[[x, y],[x+w, y+h]],"group_id": None,"shape_type":"rectangle","flags": {}}
            shapes.append(dict(dict_shape_contour))
    #=======================================
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        dict_shape_detect = {"label":label,"points":[[x0, y0],[x1, y1]],"group_id": None,"shape_type":"rectangle","flags": {}}
        shapes.append(dict(dict_shape_detect))
    #print(shapes)
    #save("frame"+str(frame_count)+ ".json","4.0.0",shapes,"frame"+str(frame_count)+ ".jpg",640,480)
    return cv2_im

if __name__ == '__main__':
    main()
