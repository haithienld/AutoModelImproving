#pip install zmq
import cv2
import zmq
import base64
import numpy as np

context = zmq.Context(2)
context1 = zmq.Context(1)
footage_socket = context.socket(zmq.SUB)
footage_socket1 = context1.socket(zmq.SUB)
footage_socket.bind('tcp://*:4664')
footage_socket1.bind('tcp://*:4665')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
footage_socket1.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

while True:
    try:
        print("\n\nBye bye\n")
        frame = footage_socket.recv_string()

        frame1 = footage_socket1.recv_string()
        
        img = base64.b64decode(frame)
        img1 = base64.b64decode(frame1)
        
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        npimg1 = np.fromstring(img1, dtype=np.uint8)
        source1 = cv2.imdecode(npimg1, 1)
        frm = np.hstack((source,source1))
        #cv2.imshow("Stream", source)
        cv2.imshow("Stream", frm)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n\nBye bye\n")
        break

