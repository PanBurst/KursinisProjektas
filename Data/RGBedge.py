import os
import cv2
import numpy as np
import cv2 as cv

baseDir = "./Data"
fileType = ".jpg"
newBaseDir = "./DataRGBedge"


def fileDFS(rootPath: str):
    currentDir = os.listdir(rootPath)
    for fileName in currentDir:
        if (fileType in fileName):
            TransformAndSave(rootPath, fileName)
        else:
            if not os.path.isfile(rootPath + "\\" + fileName):
                fileDFS(rootPath + "\\" + fileName)


def CannyImage(img, radius=10):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    w = mask.shape[0]
    h = mask.shape[1]

    _thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 25)
    _edges = cv.Canny(_thresh, 50, 200)
    for x in range(w):
        for y in range(h):
            if _edges[y][x] == 255:
                p1x, p1y, p2x, p2y = 0, 0, 0, 0
                if x < radius:
                    p1x = 0
                else:
                    p1x = x-radius
                if y < radius:
                    p1y = 0
                else:
                    p1y = y - radius

                if x+radius > w:
                    p2x = w-1
                else:
                    p2x = x+radius

                if y+radius > h:
                    p2y = h - 1
                else:
                    p2y = y+radius
                cv2.rectangle(mask, (p1x, p1y), (p2x, p2y), 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


def ThresholdImage(img, radius=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    w = mask.shape[0]
    h = mask.shape[1]
    _thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 25)
    for x in range(w):
        for y in range(h):
            if _thresh[y][x] == 0:
                p1x, p1y, p2x, p2y = 0, 0, 0, 0
                if x < radius:
                    p1x = 0
                else:
                    p1x = x-radius
                if y < radius:
                    p1y = 0
                else:
                    p1y = y - radius

                if x+radius > w:
                    p2x = w-1
                else:
                    p2x = x+radius

                if y+radius > h:
                    p2y = h - 1
                else:
                    p2y = y+radius
                cv2.rectangle(mask, (p1x, p1y), (p2x, p2y), 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    return masked


def TransformAndSave(path: str, fileName: str):
    newpath = path.replace(baseDir, newBaseDir)
    fullFilePath = path + "\\" + fileName
    fullNewFilePath = newpath + "\\" + fileName
    img = cv2.imread(fullFilePath)
    masked = CannyImage(img, 10)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    cv2.imwrite(fullNewFilePath, masked)


if __name__ == "__main__":
    fileDFS(baseDir)
