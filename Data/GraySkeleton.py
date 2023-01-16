import os
import cv2
import numpy as np
import cv2 as cv

baseDir = "./Data"
fileType = ".jpg"
newBaseDir = "./DataGraySkeleton"


def fileDFS(rootPath: str):
    currentDir = os.listdir(rootPath)
    for fileName in currentDir:
        if (fileType in fileName):
            TransformAndSave(rootPath, fileName)
        else:
            if not os.path.isfile(rootPath + "\\" + fileName):
                fileDFS(rootPath + "\\" + fileName)


def TransformAndSave(path: str, fileName: str):
    newpath = path.replace(baseDir, newBaseDir)
    fullFilePath = path + "\\" + fileName
    fullNewFilePath = newpath + "\\" + fileName
    img = cv2.imread(fullFilePath)
    masked = morphSkeleton(img, 7)
    if path.find("\\4") != -1:
        tarck = True
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    cv2.imwrite(fullNewFilePath, masked)


def morphSkeleton(img, radius=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    skel = np.zeros(gray.shape, np.uint8)

    ret, _img = cv2.threshold(gray, 75, 255, 3)
    size = np.size(_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    count = 500
    while (not done):
        eroded = cv2.erode(_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        _img = eroded.copy()

        zeros = size - cv2.countNonZero(_img)
        if zeros == size:
            done = True
        count -= 1
        if count == 0:
            break

    return skel


if __name__ == "__main__":
    fileDFS(baseDir)
