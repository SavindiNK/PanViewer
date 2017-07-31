import cv2
import time
import threading

def viewCamera(cameraID, windowName):
    cap = cv2.VideoCapture(cameraID)
    while True:
        ret, frame = cap.read()
        if ret is True:
            cv2.imshow(windowName, frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
try:
    threading.Thread(target=viewCamera, args=(0, 'Camera 1',)).start()
    time.sleep(0.5)
    threading.Thread(target=viewCamera, args=(1, 'Camera 2',)).start()
    time.sleep(0.5)
except Exception as inst:
    print type(inst)
    print inst.args
    print inst