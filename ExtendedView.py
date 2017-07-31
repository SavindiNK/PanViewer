###############################################################################
# Panorama stitching using two images
# Step 1 -  Detect keypoints and extract local invariant descriptors from the
#           two input images
# Step 2 -  Match the descriptors between the two images
# Step 3 -  Use the RANSAC algorithm to estimate a homography matrix using our
#           matched feature vectors
# Step 4 -  Apply a warping transformation using the homography matrix
###############################################################################

import cv2
import numpy as np
import time
import imutils

# ==============================================================================

# Step 1 - Detect keypoints
def detectAndDescribe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect keypoints in the image
    detector = cv2.FeatureDetector_create("SIFT")
    kps = detector.detect(gray)

    # extract features from the image
    extractor = cv2.DescriptorExtractor_create("SIFT")
    (kps, features) = extractor.compute(gray, kps)

    # convert keypoints from KeyPoint objects to NumPy arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)


# ==============================================================================

# Step 2 - Match the descriptors between the two images
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, repojThresh):
    # compute the raw matches and intialize the list of actual matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    # k=2 is used to indicate that the top two matches for each feature vector are returned

    # Lowe's ratio test
    # there can be false matches returned from the matching algorithm
    # using lowe's ratio test we select only the matches within a certain threshold
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # Step 3 - Use the RANSAC algorithm to estimate a homography matrix
    # computing the homography matrix requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, repojThresh)

        # return the matches, homography matrix and status of each matched point
        return (matches, H, status)
    return None

# ==============================================================================

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successful
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    return vis

# ==============================================================================

def stitch(imageA, imageB, ratio=0.75, reprojThresh=4.0, showMatches=False):
    # detect and extract keypoints and descriptors
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)

    # match features between the two images
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    if M is None:
        return None

    # Step 4 - Apply a perspective warp to stitch the images
    (matches, H, status) = M



    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    print len(matches)

    if showMatches:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)

    return result

# ==============================================================================

# adjusts brighgtness of the images
def equalizeImage(img):
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #intensity is given by the Y channel
    #equalize the histogram of the Y channel
    imgYUV[:,:,0] = cv2.equalizeHist(imgYUV[:,:,0])
    imgEqualized = cv2.cvtColor(imgYUV, cv2.COLOR_YUV2BGR)
    imgEqualized[0,:,:] = cv2.equalizeHist(imgEqualized[0,:,:])
    imgEqualized[:,0,:] = cv2.equalizeHist(imgEqualized[:,0,:])
    imgEqualized[:,:,0] = cv2.equalizeHist(imgEqualized[:,:,0])
    return imgEqualized

#===============================================================================

# initializes cameras - have to handle exceptions
def initCameras():
    global cap1, cap2
    # initialize cameras
    cap1 = cv2.VideoCapture(1)
    time.sleep(1)
    cap2 = cv2.VideoCapture(0)
    time.sleep(1)

#===============================================================================

# returns homography matrix for given images. key point matching is done inside
# this method
def getHomographyMatrix(frameA, frameB):
    global ratio, reprojThresh
    (kpsA, featuresA) = detectAndDescribe(frameA)
    (kpsB, featuresB) = detectAndDescribe(frameB)

    # match features between the two images
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    if M is None:
        return None

    # Step 4 - Apply a perspective warp to stitch the images
    (matches, H, status) = M
    return H

# ==============================================================================
# ==============================================================================
ratio = 0.75
reprojThresh = 4.0

cap1 = 1
cap2 = 0
initCameras()

# take initial images to calibrate
retA, frameA = cap1.read()
# cv2.imshow('A', frameA)

retB, frameB = cap2.read()
# cv2.imshow('B', frameB)

H = getHomographyMatrix(frameA, frameB)

while True:
    ret1, frame_1 = cap1.read()
    ret2, frame_2 = cap2.read()
    result = cv2.warpPerspective(equalizeImage(frame_1), H, (frame_1.shape[1] + frame_2.shape[1], frame_1.shape[0]))
    result[0:frame_2.shape[0], 0:frame_2.shape[1]] = equalizeImage(frame_2)

    if ret1 and ret2:
        cv2.imshow('Test_1_RT', result)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
cap1.release()
cap2.release()