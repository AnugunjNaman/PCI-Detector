import os
import cv2

# full path to the video file
path = 'G:/open cv/japan_road.mp4'
# full path to the output directory
out_path = 'G:/open cv/imgs/'
# How many frames to skip after taking a image
SKIP_FRAME = 20

cap =cv2.VideoCapture(path)
c = 0
while True:
    ret,frame = cap.read()
    c +=1
    resized = cv2.resize(frame,(600,600),interpolation=cv2.INTER_LINEAR)
    if c %SKIP_FRAME==0:
        # print((out_path+'{}.jpg'.format(str(c))))
        cv2.imwrite(out_path+'custom_'+'{}.jpg'.format(str(c)),resized)
        # print('...File Written...')
cap.release()
cv2.destroyAllWindows()
