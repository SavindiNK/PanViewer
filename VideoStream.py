from threading import Thread
from Calibrator import Distort_Remover
import cv2
import time

class VideoStream:
    def __init__(self, cb, src=0):
        # initialize the camera and read the first frame
        self.stream = cv2.VideoCapture(src)
        time.sleep(1)
        (self.grabbed, self.frame) = self.stream.read()
        self.distort_remover = Distort_Remover(cb)
        self.undistorted = None
        # if self.grabbed:
        #     self.undistorted = self.distort_remover.undistort(self.frame)
        # indicator variables
        self.stopped = False
        self.id = src

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            # if self.grabbed:
            #     self.undistorted = self.distort_remover.undistort(self.frame)

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

