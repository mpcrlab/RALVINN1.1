__author__ = 'mpcr'

import numpy as np
import cv2

class RoverImageProcessor:
    def __init__(self):
        self.masks = {}
        self.colors = {
            "orange": [np.array([9, 100, 100]), np.array([17,256,256])],
            "pink": [np.array([145, 100, 100]), np.array([178, 256, 256])]
        }

    #function that adds a color to the dictionary of colors
    #color range parameters are lists in the form [h, s, v]
    def addColor(self, name, lower_hsv_range, upper_hsv_range):
        self.colors[name] = [np.array(lower_hsv_range), np.array(upper_hsv_range)]

    #image is an opencv image to be processed
    #colors is a list of colors to check for.
    #each color in the list is a plain english string e.g. "orange"
    def process(self, image, colors, minCntArea = 1000):
        img = image
        imgHeight, imgWidth, imgChannels = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #states is the object to be returned
        #it contains a list of ints - one for each color
        #each int specifies the quadrant that color was found
        #0 - not found
        #1 - left
        #2 - center
        #3 - right
        states = []

        for color in colors:
            if color not in self.masks:
                self.masks[color] = []
            mask = cv2.inRange(hsv, self.colors[color][0], self.colors[color][1])
            res = cv2.bitwise_and(img, img, mask=mask)
            self.masks[color].append(res)
            if len(self.masks[color]) > 3:
                mask1, mask2, mask3 = self.masks[color][-1],self.masks[color][-2],self.masks[color][-3]
                mask2 = cv2.bitwise_and(mask1, mask2)
                res = cv2.bitwise_and(mask2, mask3)
                del self.masks[color][0]
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            cContours, cHierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(cContours) == 0:
                states.append(0)
            else:
                cnt = []
                M = []
                highestArea = 0
                biggestCnt = -1
                for i in xrange(len(cContours)):
                    cnt.append(cContours[i])
                    M.append(cv2.moments(cnt[i]))
                    cntArea = cv2.contourArea(cnt[i])
                    if cntArea > highestArea and cntArea > minCntArea:
                        highestArea = cntArea
                        biggestCnt = i
                if biggestCnt == -1:
                    states.append(0)
                else:
                    try:
                        cx = int(M[biggestCnt]['m10']/M[biggestCnt]['m00'])
                        #cy = int(M[biggestCnt]['m01']/M[biggestCnt]['m00'])
                        if cx <= imgWidth / 3:
                            states.append(1)
                        elif cx > 2* imgWidth / 3:
                            states.append(3)
                        else:
                            states.append(2)
                    except:
                        states.append(0)


            #these lines only for debugging - display the contours found for each color
            display = res[:,:].copy()
            cv2.imshow(color, display)
            cv2.waitKey(1)


        return states