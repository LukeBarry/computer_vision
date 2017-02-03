import cv2
import numpy as np
from PIL import Image
import os

# Download goo.gl/nXaoEf as owl.png
owl = cv2.imread('C:/Users/lbarry/Desktop/20.png')
print owl.shape # (140, 160, 3)
#cv2.imshow('owl', owl)
owl_gray = cv2.cvtColor(owl, cv2.COLOR_BGR2GRAY)
print owl_gray.shape # (140, 160)
#cv2.imshow('owl gray', owl_gray)
cv2.waitKey()

owl_gray_mask = owl_gray > 0
owl_mask = np.array(
    [[[v, v, v] for v in row] for row in owl_gray_mask]
    )

pilowl = Image.open('C:/Users/lbarry/Desktop/20.png')
# PIL -> cv2
owl2 = np.asarray(pilowl)
owl2 = cv2.cvtColor(owl2, cv2.COLOR_RGB2BGR)
# cv2 -> PIL
pilowl2 = cv2.cvtColor(owl2, cv2.COLOR_BGR2RGB)
pilowl2 = Image.fromarray(pilowl2)


# Load face detection cascade
print 'loading face cascade...'
# cas = cv2.CascadeClassifier('C:\Users\lbarry\computer_vision\frontalface.xml')
cas_dir = 'C:/Users/lbarry/Downloads/opencv/sources/data/haarcascades_cuda/'
cas_fname = 'haarcascade_frontalface_default.xml'
cas = cv2.CascadeClassifier(os.path.join(cas_dir, cas_fname))

print 'starting webcam...'
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
assert ret # fails if couldn't read from webcam
print 'webcam res:', frame.shape

while True:
    ret, frame = cam.read()
    # overlay the owl
    # np.copyto(frame[-owl.shape[0]:, 0:owl.shape[1], :], owl, where=owl_mask)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cas.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        #flags = cv2.cv.CASCADE_SCALE_IMAGE
        # OpenCV 2.4 -> cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    # faces is list of (x, y, width, height)                   
                             
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # this is commented out on github
        # Make the box a bit bigger (see why later)
        #- first save originals
        x0 = x; y0 = y; w0 = w; h0 = h
        h += int(h * 0.0)
        x -= int(w * 0.0)
        w += int(w * 0.0)
        if y < 0 or y+h > frame.shape[0]: continue
        if x < 0 or x+w > frame.shape[1]: continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2) # this is commented out on github
 
              
        # Put owl in box
        hair = owl
        hair_mask = owl_mask        
        hair_scale = w0 / float(hair.shape[1])      
        hair_mask = np.array(hair_mask, dtype='uint8')
        hair_mask = np.array(hair_mask, dtype=bool)
        hair = np.array(np.minimum((hair * 0.7) + 100, 255), dtype='uint8') # Optional

        #if y0 - hair.shape[0] < 0:
        if y0 + hair.shape[0] >= frame.shape[0]: 
            continue
        #It's got to be this line that puts the owl on top    
        np.copyto(frame[y0:y0 + hair.shape[0], x0:x0+hair.shape[1], :], hair, where=hair_mask)
                                    
    cv2.imshow('cam', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break

print faces
print 'webcam res:', frame.shape