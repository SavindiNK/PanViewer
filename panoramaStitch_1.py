# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 19:43:01 2017

@author: Savindi Niranthara
"""

## Testing 1

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

cv2.ocl.setUseOpenCL(False)

###############################################################################

def drawMatches(imgA,kpA,imgB,kpB,matches,status):
    (hA, wA) = imgA.shape[:2]
    (hB, wB) = imgB.shape[:2]
    vis = np.zeros((max(hA,hB), wA+wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imgA
    vis[0:hB, wA:] = imgB
    
    #loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
    		# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
				ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    # return the visualization
    return vis

###############################################################################

def detectAndDescribe(img):
    #find keypoints using ORB
    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(img, None)
    #convert keypoints into a numpy array
    kpArray = np.float32([k.pt for k in kp])
    return (kpArray,desc)

###############################################################################

def matchKeypoints(kpArrayA, descA, kpArrayB, descB, ratio, reprojThresh):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(descA,descB,k=2)
    matches = []
    
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance*ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    
    if len(matches) > 4:
        #construct the two sets of points
        ptsA = np.float32([kpArrayA[i] for (_,i) in matches])
        ptsB = np.float32([kpArrayB[i] for (i,_) in matches])
    
        #compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,reprojThresh)
    
        #return matches, homography matrix and status of each point
        return (matches, H, status)
    return None

###############################################################################

def stitch(img1, img2, ratio=0.75, reprojThresh=4.0, showMatches=False):
    (kpArray1, desc1) = detectAndDescribe(img1)
    (kpArray2, desc2) = detectAndDescribe(img2)
    
    M = matchKeypoints(kpArray1,desc1,kpArray2,desc2,ratio,reprojThresh)
    
    if M is None:
        return None
    
    (matches, H, status) = M
    result = cv2.warpPerspective(img1,H,(img1.shape[1]+img2.shape[1],img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    if showMatches:
        vis = drawMatches(img1,kpArray1,img2,kpArray2,matches,status)
        return (result, vis)
    return result

###############################################################################

#read the two images
#img1 = cv2.imread('sedona2.png', 0)
#img2 = cv2.imread('sedona1.png', 0)

img1 = cv2.imread('LS4.jpg', cv2.CAP_MODE_RGB)
img2 = cv2.imread('LS3.jpg', cv2.CAP_MODE_RGB)

img1 = imutils.resize(img1, width=400)
img2 = imutils.resize(img2, width=400)

(result, vis) = stitch(img1,img2,showMatches=True)

plt.imshow(img1),plt.show()
plt.imshow(img2),plt.show()
plt.imshow(vis),plt.show()

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
cv2.imwrite("results.jpg", result)