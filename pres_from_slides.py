import cv2
import numpy as np
from PIL import Image
import os

# Download goo.gl/nXaoEf as squirrel.png
sq = cv2.imread('C:\Users\lbarry\Downloads\squirrel.png')
#print sq.shape # (140, 160, 3)
#cv2.imshow('squirrel', sq)
sq_gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
#print sq_gray.shape # (140, 160)
#cv2.imshow('squirrel gray', sq_gray)
cv2.waitKey()

sq_gray_mask = sq_gray > 0
sq_mask = np.array(
    [[[v, v, v] for v in row] for row in sq_gray_mask]
    )

pilsq = Image.open('C:\Users\lbarry\Downloads\squirrel.png')
# PIL -> cv2
sq2 = np.asarray(pilsq)
sq2 = cv2.cvtColor(sq2, cv2.COLOR_RGB2BGR)
# cv2 -> PIL
pilsq2 = cv2.cvtColor(sq2, cv2.COLOR_BGR2RGB)
pilsq2 = Image.fromarray(pilsq2)


# Load face detection cascade
print 'loading face cascade...'
cas = cv2.CascadeClassifier('C:\Users\lbarry\computer_vision\frontalface.xml')
#cas_dir = '/usr/share/opencv/haarcascades/'
#cas_fname = 'haarcascade_frontalface_default.xml' 
#cas = cv2.CascadeClassifier(os.path.join(cas_dir, cas_fname))

print 'starting webcam...'
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
assert ret # fails if couldn't read from webcam
print 'webcam res:', frame.shape

while True:
    ret, frame = cam.read()
    # overlay the squirrel
    np.copyto(frame[-sq.shape[0]:, 0:sq.shape[1], :], sq, where=sq_mask)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cas.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        # OpenCV 2.4 -> cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    # faces is list of (x, y, width, height)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # this is commented out on github
        # Make the box a bit bigger (see why later)
        #- first save originals
        x0 = x; y0 = y; w0 = w; h0 = h
        h += int(h * 0.3)
        x -= int(w * 0.1)
        w += int(w * 0.2)
        if y < 0 or y+h > frame.shape[0]: continue
        if x < 0 or x+w > frame.shape[1]: continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2) # this is commented out on github
        
    cv2.imshow('cam', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break




