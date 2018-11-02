
import numpy as np
import cv2 as cv
# from PIL import Image
import matplotlib.pyplot as plt


# def ppm2jpg(dirname):
#     for i in range(6):
#         fname = './' + str(dirname) + '/img' + str(i+1) + '.ppm'
#         img = Image.open(fname)
#         img.save('./' + str(dirname + '/img' + str(i+1) + '.jpg'))


class Stitcher(object):

    def __init__(self):
        self.MIN_MATCH_COUNT = 4
        pass

    def Stitch(self, imgs, ratio=0.75, reprojThresh=5.0, showMatches=False):

        imgA, imgB = imgs
        kpA, desA = self.detectKP_and_descripe(imgA)
        kpB, desB = self.detectKP_and_descripe(imgB)

        M = self.matchKP(kpA, desA, kpB, desB, ratio, reprojThresh)

        if M is None:
            return None

        matches, H, status = M

        result = cv.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
        result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB

        if showMatches:
            matchesMask = status.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            imgShowMatches = cv.drawMatches(imgA, kpA, imgB, kpB, matches, None, **draw_params)

            return result, imgShowMatches

        return result


    def detectKP_and_descripe(self, img):

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        return kp, des


    def matchKP(self, kpA, desA, kpB, desB, ratio, reprojThresh):

        bf = cv.BFMatcher()
        rawMatches = bf.knnMatch(desA, desB, k=2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append(m[0])

        # result = cv.drawMatchesKnn(imgA, kpA, imgB, kpB, matches, outImg=None, flags=2)

        if len(matches) > self.MIN_MATCH_COUNT:
            ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)
            return matches, M, mask

        return None


if __name__ == '__main__':
    # imgA = cv.imread('classroom1.jpg')
    # imgB = cv.imread('classroom2.jpg')
    imgA = cv.imread('img1.jpg')
    imgB = cv.imread('img2.jpg')
    Stitcher = Stitcher()
    result, imgShowMatches = Stitcher.Stitch([imgB, imgA], showMatches=True)

    cv.imshow('imgA', imgA)
    cv.imshow('imgB', imgB)
    cv.imshow('result', result)
    cv.imshow('imgMatcher', imgShowMatches)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('stitch_mt.jpg', result)
    cv.imwrite('matches_mt.jpg', imgShowMatches)