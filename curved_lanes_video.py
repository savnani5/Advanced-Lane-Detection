import numpy as np
import cv2

cap = cv2.VideoCapture("Data2/challenge_video.mp4")
counter = 0
framerate = 1  

while(cap.isOpened()):
    ret, frame = cap.read()
    width = int(frame.shape[1] * 0.5)
    height = int(frame.shape[0] * 0.5)
    frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    cv2.imwrite(f'Data2/images/image{counter}.png', frame)
    if cv2.waitKey(framerate) & 0xFF == ord('q'):
        break
    
    counter +=1

cap.release()
cv2.destroyAllWindows()



source_points = np.float32([[317.2,203.6], [370.9,203.2], [470.4,291.6], [160.3, 291.6]])
dst_points = np.float32([[160.3, 111.6], [470.4,111.6], [470.4,291.6], [160.3, 291.6]])