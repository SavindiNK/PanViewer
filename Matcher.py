import numpy as np
import cv2


class Matcher:
    def __init__(self):
        self.detector = cv2.FeatureDetector_create("SIFT")
        self.extractor = cv2.DescriptorExtractor_create("SIFT")
        self.matcher = cv2.DescriptorMatcher_create("BruteForce")

    # Step 1 - Detect keypoints
    def detectAndDescribe(self,image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect keypoints in the image
        kps = self.detector.detect(gray)

        # extract features from the image
        (kps, features) = self.extractor.compute(gray, kps)

        # convert keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    # Step 2 - Match the descriptors between the two images
    def matchKeypoints(self,kpsA, kpsB, featuresA, featuresB, ratio, repojThresh):
        # compute the raw matches and intialize the list of actual matches
        rawMatches = self.matcher.knnMatch(featuresA, featuresB, 2)
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

    def match(self, imageA, imageB, ratio=0.75, reprojThresh=4.0):
        # detect and extract keypoints and descriptors
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            return None

        # Step 4 - Apply a perspective warp to stitch the images
        (matches, H, status) = M
        return H