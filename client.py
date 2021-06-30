#pip install zmq
import cv2
import zmq
import base64
import numpy as np
from PIL import Image
'''
context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:4664')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

while True:
    try:
        frame = footage_socket.recv_string()    
        img = base64.b64decode(frame)
        
        npimg = np.fromstring(img, dtype=np.uint8)
        frame1 = footage_socket.recv_json() 
        #a,b = npimg[0],npimg[1]
        print("image",npimg)
        print("json",frame1)
        #break
        #print(a)
        #print(b)
        source = cv2.imdecode(npimg, 1)
        
        cv2.imshow("Stream", source)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n\nBye bye\n")
        break
'''
def recv_array_and_str(socket, flags=0, copy=True, track=False):
    string = socket.recv_string(flags=flags)
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)

    img = np.frombuffer(bytes(memoryview(msg)), dtype=md['dtype'])
    return string, img.reshape(md['shape'])

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:4664')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
while True:
    msg, img = recv_array_and_str(footage_socket)
    print(msg)
    im_arr = np.array(img) 
    #img = Image.fromarray(im_arr, 'RGB')
    print(im_arr)
    cv2.imshow("Stream", im_arr)

