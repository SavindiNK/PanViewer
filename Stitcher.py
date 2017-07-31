import numpy as np
import cv2
import imutils
from Matcher import Matcher


class Stitcher:
    def __init__(self, frames_list, Hs):
        self.images = frames_list
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = Matcher()
        self.prepare_lists()
        self.Hs = Hs

    def prepare_lists(self):
        self.centerIdx = self.count - 1
        self.center_im = self.images[self.centerIdx]
        for i in range(self.count):
            if (i <= self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

    def leftshift(self):
        a = self.left_list[0]
        tmp = None
        for i in range(1, len(self.left_list)):
            b = self.left_list[i]
            H = self.Hs[i-1]
            # since we are stitching backwards we need the inverse
            xh = np.linalg.inv(H)

            # to calculate the dimension
            f1 = np.dot(xh, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)

            # now warp
            tmp = cv2.warpPerspective(a, xh, dsize)
            print offsetx, offsety

            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
            a = tmp

        self.result = tmp

    def rightshift(self):
        for each in self.right_list:
            H = self.matcher_obj.match(each, self.leftImage)

            # # dimension
            # f1 = np.dot(H, np.array([0, 0, 1]))
            # f1 = f1 / f1[-1]
            # H[0][-1] += abs(f1[0])
            # H[1][-1] += abs(f1[1])
            # ds = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            # offsety = abs(int(f1[1]))
            # offsetx = abs(int(f1[0]))
            # dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)

            # to calculate dimensions of the warped image
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]
            dsize = (int(txyz[0]) + self.leftImage.shape[1], int(txyz[1]) + self.leftImage.shape[0])

            # now warp
            tmp = cv2.warpPerspective(each, H, dsize)
            # cv2.imshow('X', tmp)
            # cv2.waitKey(0)

            # tmp[offsety:self.leftImage.shape[0] + offsety, offsetx:self.leftImage.shape[1] + offsetx] = self.leftImage

            # # to overlap
            # self.leftImage[self.leftImage.shape[0]:int(txyz[1]) + self.leftImage.shape[0],
            # self.leftImage.shape[1]:int(txyz[0]) + self.leftImage.shape[1]] = tmp

    def stitch_all(self):
        img1 = self.images[0]
        result = None
        for i in range(1,len(self.images)):
            img2 = self.images[i]
            result = cv2.warpPerspective(img1, self.Hs[i-1], (img1.shape[1] + img2.shape[1], img1.shape[0]))
            result[0:img2.shape[0], 0:img2.shape[1]] = img2
            img1 = result
        self.result = result

    def set_images(self, images):
        self.images = images
        self.left_list = images

#
# img1 = cv2.imread('./images/WC_1.jpg')
# img1 = imutils.resize(img1, width=400)
# img2 = cv2.imread('./images/WC_2.jpg')
# img2 = imutils.resize(img2, width=400)
# img3 = cv2.imread('./images/WC_3.jpg')
# img3 = imutils.resize(img3, width=400)
# img4 = cv2.imread('./images/WC_4.jpg')
# img4 = imutils.resize(img3, width=400)
#
# Hs = []
#
# stitcher  = Stitcher([img1, img2, img3, img4], None)
# stitcher.leftshift()
# cv2.imshow(' Result',stitcher.leftImage)
# cv2.waitKey(0)