import numpy as np
import time
import cv2


def drawMatches(img1, kp1, img2, kp2, matches):

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def detectAndDescribe(image):
    sift = cv2.FeatureDetector_create('SIFT')
    kps = sift.detect(image)

    extractor = cv2.DescriptorExtractor_create("SIFT")
    (kps, features) = extractor.compute(image, kps)
    return (kps, features)

def match(featureA, featureB):
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checcks=50)

    matcher = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = matcher.knnMatch(featuresA, featuresB, k=2)
    return matches

def refineMatches(matches, reprojThresh=0.75):
    refinedMatches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * reprojThresh:
            refinedMatches.append((m[0].trainIdx, m[0].queryIdx))
    return refinedMatches

def findHomo(refinedMatches, reprojThresh=0.75):
    if len(refinedMatches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i].pt for (_, i) in refinedMatches])
        ptsB = np.float32([kpsB[i].pt for (i, _) in refinedMatches])
        return cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
    else:
        return None

def warp(frameA, frameB, H):
    result = cv2.warpPerspective(frameA, H, (frameA.shape[1] + frameB.shape[1], frameB.shape[0]))
    result[0:frameB.shape[0], 0:frameB.shape[1]] = frameB
    return result

# initialize cameras
cap1 = cv2.VideoCapture(1)
time.sleep(1)
cap2 = cv2.VideoCapture(0)
time.sleep(1)

# take initial images to calibrate
retA, frameA = cap1.read()
cv2.imshow('A', frameA)

retB, frameB = cap2.read()
cv2.imshow('B', frameB)

(kpsA, featuresA) = detectAndDescribe(frameA)
(kpsB, featuresB) = detectAndDescribe(frameB)

matches = match(featuresA, featuresB)

refinedMatches = refineMatches(matches)

(H, status) = findHomo(refinedMatches)

result = warp(frameA, frameB, H)

cv2.imshow('Test_1', result)
cv2.imwrite('result_conda.jpg', result)
cv2.waitKey(0)

while True:
    ret1, frame_1 = cap1.read()
    ret2, frame_2 = cap2.read()
    result_1 = warp(frame_1, frame_2, H)

    if ret1 and ret2:
        cv2.imshow('Test_1_RT', result_1)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap1.release()
cap2.release()