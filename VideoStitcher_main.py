from Calibrator import Calibrator
from VideoStream import VideoStream
from Matcher import Matcher
from Stitcher import Stitcher
import time
import cv2


# obtain calibrate matrices
calibrator = Calibrator()
calibrator.calibrate()
matcher = Matcher()


# initialize video streams
no_of_streams = 2
vss = [VideoStream(calibrator, src=1), VideoStream(calibrator, src=0)]
calibrator.calculate_optimal_camera_matrix(vss[0].read().shape[1],vss[0].read().shape[0])

# initialize homographies
homographies = []
for i in range(no_of_streams - 1):
    homographies.append(matcher.match(vss[i+1].frame, vss[i].frame))

vss_frames_list = []
for i in range(no_of_streams):
    vss_frames_list.append(vss[i].read())

stitcher = Stitcher(vss_frames_list, homographies)

vss[0].start()
time.sleep(1)
vss[1].start()
time.sleep(1)

while True:
    frame1 = vss[0].frame
    # print frame1
    frame2 = vss[1].frame

    stitcher.set_images([frame1, frame2])
    stitcher.leftshift()

    cv2.imshow('Result', stitcher.result)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vss[0].stop()
vss[1].stop()

# stitching

#   apply homographies
#       warp ordering
#       recursive warping
# apply smoothing and feathering