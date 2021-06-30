#pip install zmq
import cv2
import zmq
import base64
import numpy as np
import time
from datetime import datetime
import subprocess as sp

jpeg_frames = []

context = zmq.Context()

footage_socket = context.socket(zmq.SUB)

footage_socket.bind('tcp://*:4664')

footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

#out = cv2.VideoWriter(args[0].stream_out, cv2.VideoWriter_fourcc(*args[0].encoding_codec),
#                                           int(self.cap.get(cv2.CAP_PROP_FPS)), (640, 480))

out = cv2.VideoWriter(str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))+'.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
fh = open("video.mp4", "wb")
while True:
    try:
        print("\n\nBye bye\n")
        frame = footage_socket.recv_string()

        
        img = base64.b64decode(frame)
        
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        
        out.write(source)
        
        cv2_img = cv2.resize(source,(1280,960))
        cv2.imshow("Stream", cv2_img)
        #cv2.imwrite("reconstructed.jpg", cv2_img)

        
        
        
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n\nBye bye\n")
        break

