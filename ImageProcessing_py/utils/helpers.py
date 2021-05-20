import os
import matplotlib.pyplot as plt
import math
import imutils

from ImageProcessing_py.utils.matching import *
from ImageProcessing_py.utils.combine import *

logger = logging.getLogger("main")


def is_cv2():
    major, minor, increment = cv2.__version__.split(".")
    return major == "2"


def is_cv3():
    major, minor, increment = cv2.__version__.split(".")
    return major == "3"

def is_cv4():
    major, minor, increment = cv2.__version__.split(".")
    return major == "4"


def display(title, img, max_size=500000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)

def displayCor(title, img, location, max_size=500000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    img1 = img.copy()
    scale = numpy.sqrt(min(1.0, float(max_size) / (img1.shape[0] * img1.shape[1])))
    shape = (int(scale * img1.shape[1]), int(scale * img1.shape[0]))
    img1 = cv2.resize(img1, shape)
    k1, k2, k3 = 0.5, 0.2, 0.0  # 배럴 왜곡
    #k1, k2, k3 = -0.3, 0, 0    # 핀큐션 왜곡

    rows, cols = img1.shape[:2]

    # 매핑 배열 생성 ---②
    mapy, mapx = numpy.indices((rows, cols), dtype=numpy.float32)

    # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환 ---③
    mapx = 2 * mapx / (cols - 1) - 1
    mapy = 2 * mapy / (rows - 1) - 1
    r, theta = cv2.cartToPolar(mapx, mapy)

    # 방사 왜곡 변영 연산 ---④
    ru = numpy.linalg.inv(r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6)))


    # 직교좌표 및 좌상단 기준으로 복원 ---⑤
    mapx, mapy = cv2.polarToCart(ru, theta)
    mapx = ((mapx + 1) * cols - 1) / 2
    mapy = ((mapy + 1) * rows - 1) / 2
    # 리매핑 ---⑥
    distored = cv2.remap(img1, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imshow(title, distored)

def display_red(title, img, corner, max_size=500000):
    #assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    #assert isinstance(title, str), 'title must be a string'
    img1 = img.copy()
    cv2.drawContours(img1, [corner], 0, (0, 0, 255), 1)
    scale = numpy.sqrt(min(1.0, float(max_size) / (img1.shape[0] * img1.shape[1])))
    shape = (int(scale * img1.shape[1]), int(scale * img1.shape[0]))

    #img1 = cv2.resize(img1, shape)
    cv2.imshow(title, img1)

    return img1

def save_image(path, result):
    name, ext = os.path.splitext(path)
    img_path = '{0}.png'.format(name)
    logger.debug('writing image to {0}'.format(img_path))
    cv2.imwrite(img_path, result)
    logger.debug('writing complete')

def removeMargin(result):
    # 마스크 구하기(픽셀값 0,255로 나누기)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1])
    thresh = cv2.medianBlur(thresh, 5)

    plt.figure(figsize=(20, 20))
    plt.imshow(thresh, cmap='gray')

    # 테두리 지우기
    stitched_copy = result.copy()
    thresh_copy = thresh.copy()

    while numpy.sum(thresh_copy) > 0:
        thresh_copy = thresh_copy[1:-1, 1:-1]
        stitched_copy = stitched_copy[1:-1, 1:-1]

    return stitched_copy

def ifExist(result):
    result1 = result.copy()
    gray = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1])
    thresh = cv2.medianBlur(thresh, 5)
    thresh_copy = thresh.copy()
    return thresh_copy

def removeMarginLeast(frame, pixel):
    result1 = frame.copy()
    gray = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1])
    thresh = cv2.medianBlur(thresh, 5)
    count, minP, maxP = 0

    for i in range(0, thresh.shape[0]):
        for j in range(0, thresh.shape[1]):
            if thresh[i, j] != 255:
                minP = j
                count += 1
                break
        if count > 0:
            thresh[i, minP:minP + pixel] = 0
        count = 0

    for i in range(0, thresh.shape[0]):
        for j in range(thresh.shape[1] - 1, 0, -1):
            if thresh[i, j] != 255:
                maxP = j
                count += 1
                break
        if count > 0:
            thresh[i, maxP - pixel:maxP] = 0
        count = 0

    for i in range(0, thresh.shape[1]):
        for j in range(0, thresh.shape[0]):
            if thresh[j, i] != 255:
                minP = j
                count += 1
                break
        if count > 0:
            thresh[minP:minP + pixel, i] = 0
        count = 0

    for i in range(0, thresh.shape[1]):
        for j in range(thresh.shape[0] - 1, 0, -1):
            if thresh[j, i] != 255:
                maxP = j
                count += 1
                break
        if count > 0:
            thresh[maxP - pixel:maxP, i] = 0
        count = 0

    return thresh

def rvMar_retnP(frame, rm_pixel, bl_pixel, bunza, bunmo):
    # return removed margin(rm_pixel) frame, its location
    # threshold, blending(bl_pixel) point
    result = frame.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1])
    thresh = cv2.medianBlur(thresh, 5)
    #find_contours
    count = 0
    minP, maxP, maxX, maxY = 0
    minX, minY = thresh.shape[0:2]

    blending_list = {}
    a = bunza  # 분자 a = 4, b = 5 4/5~5/5
    b = bunmo # 분모
    for i in range(0, thresh.shape[0]):
        for j in range(0, thresh.shape[1]):
            if thresh[i, j] != 255:
                minP = j
                count += 1
                break
        if count > 0:
            if minP + rm_pixel < minY:
                minY = minP + rm_pixel
            result[i, minP:minP + rm_pixel] = [0, 0, 0, 0]
            thresh[i, minP:minP + rm_pixel] = 255
            for k in range(minP + rm_pixel, minP + rm_pixel + bl_pixel):
                if thresh[i, k] == 0:
                    thresh[i, k] = 128
                    proportion = ((a*bl_pixel - 1 + k - (minP + rm_pixel)) / (bl_pixel * b))
                    location = str(i) + " " + str(k)
                    blending_list[location] = proportion
                else:
                    break
        count = 0

        for j in range(thresh.shape[1] - 1, 0, -1):
            if thresh[i, j] != 255:
                maxP = j
                count += 1
                break
        if count > 0:
            if maxP + 1 - rm_pixel > maxY:
                maxY = maxP + 1 - rm_pixel
            result[i, maxP + 1 - rm_pixel:maxP + 1] = [0, 0, 0, 0]
            thresh[i, maxP + 1 - rm_pixel:maxP + 1] = 255
            for k in range(maxP + 1 - rm_pixel - bl_pixel, maxP + 1 - rm_pixel):
                if thresh[i, k] == 0:
                    thresh[i, k] = 128
                    proportion = ((a+1)*bl_pixel - 1 - k + (maxP + 1 - rm_pixel - bl_pixel)) / (bl_pixel * b)
                    location = str(i) + " " + str(k)
                    blending_list[location] = proportion
        count = 0

    for i in range(0, thresh.shape[1]):
        for j in range(0, thresh.shape[0]):
            if thresh[j, i] != 255:
                minP = j
                count += 1
                break
        if count > 0:
            if minP + rm_pixel < minX:
                minX = minP + rm_pixel
            result[minP:minP + rm_pixel, i] = [0, 0, 0, 0]
            thresh[minP:minP + rm_pixel, i] = 255
            for k in range(minP + rm_pixel, minP + rm_pixel + bl_pixel):
                if thresh[k, i] == 0:
                    thresh[k, i] = 128
                    proportion = (a*bl_pixel - 1 + k - (minP + rm_pixel)) / (bl_pixel * b)
                    location = str(k) + " " + str(i)
                    blending_list[location] = proportion
        count = 0

        for j in range(thresh.shape[0] - 1, 0, -1):
            if thresh[j, i] != 255:
                maxP = j
                count += 1
                break
        if count > 0:
            if maxP + 1 - rm_pixel > maxX:
                maxX = maxP + 1 - rm_pixel
            result[maxP + 1 - rm_pixel:maxP + 1, i] = [0, 0, 0, 0]
            thresh[maxP + 1 - rm_pixel:maxP + 1, i] = 255
            for k in range(maxP + 1 - rm_pixel - bl_pixel, maxP + 1 - rm_pixel):
                if thresh[k, i] == 0:
                    thresh[k, i] = 128
                    proportion = ((a+1)*bl_pixel - 1 - k + (maxP + 1 - rm_pixel - bl_pixel)) / (bl_pixel * b)
                    location = str(k) + " " + str(i)
                    blending_list[location] = proportion
        count = 0

    points = [minX, minY, maxX, maxY]

    return result, thresh, points, blending_list

def cylindrical_projection(img, focal_length):
    height, width, _ = img.shape
    cylinder_proj = numpy.zeros(shape=img.shape, dtype=numpy.uint8)

    for y in range(-int(height / 2), int(height / 2)):
        for x in range(-int(width / 2), int(width / 2)):
            cylinder_x = focal_length * math.atan(x / focal_length)
            cylinder_y = focal_length * y / math.sqrt(x ** 2 + focal_length ** 2)

            cylinder_x = round(cylinder_x + width / 2)
            cylinder_y = round(cylinder_y + height / 2)

            if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                cylinder_proj[cylinder_y][cylinder_x] = img[y + int(height / 2)][x + int(width / 2)]

    # Crop black border
    # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    _, thresh = cv2.threshold(cv2.cvtColor(cylinder_proj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    return cylinder_proj[y:y + h, x:x + w]

def cylindrical_wprojection(img, focal_length):
    height, width, _ = img.shape
    cylinder_proj = numpy.zeros(shape=img.shape, dtype=numpy.uint8)

    for x in range(-int(width / 2), int(width / 2)):
        for y in range(-int(height / 2), int(height / 2)):
            cylinder_y = focal_length * math.atan(y / focal_length)
            cylinder_x = focal_length * x / math.sqrt(y ** 2 + focal_length ** 2)

            cylinder_x = round(cylinder_x + width / 2)
            cylinder_y = round(cylinder_y + height / 2)

            if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                cylinder_proj[cylinder_y][cylinder_x] = img[y + int(height / 2)][x + int(width / 2)]

    # Crop black border
    # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    _, thresh = cv2.threshold(cv2.cvtColor(cylinder_proj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    return cylinder_proj[y:y + h, x:x + w]

def medianFilter(output_img, warped_img, i, j, ratio):
    listB = []
    listG = []
    listR = []
    a = ratio
    b = 1 - ratio
    B, G, R = 0, 0, 0
    Rst = False

    if warped_img[i, j, 0] > 0 or warped_img[i, j, 1] > 0 or warped_img[i, j, 2] > 0:
        if 0 <= i - 1 <= warped_img.shape[0]:
            if 0 <= j - 1 < warped_img.shape[1]:
                listB.append(a*output_img[i - 1, j - 1, 0] + b*warped_img[i - 1, j - 1, 0])
                listG.append(a*output_img[i - 1, j - 1, 1] + b*warped_img[i - 1, j - 1, 1])
                listR.append(a*output_img[i - 1, j - 1, 2] + b*warped_img[i - 1, j - 1, 2])
            #listB.append(warped_img[i - 1, j - 1, 0])
            #listG.append(warped_img[i - 1, j - 1, 1])
            #listR.append(warped_img[i - 1, j - 1, 2])
            if 0 <= j < warped_img.shape[1]:
                listB.append(a*output_img[i - 1, j, 0] + b*warped_img[i - 1, j, 0])
                listG.append(a*output_img[i - 1, j, 1] + b*warped_img[i - 1, j, 1])
                listR.append(a*output_img[i - 1, j, 2] + b*warped_img[i - 1, j, 2])
            #listB.append(warped_img[i - 1, j, 0])
            #listG.append(warped_img[i - 1, j, 1])
            #listR.append(warped_img[i - 1, j, 2])
            if 0 <= j + 1 < warped_img.shape[1]:
                listB.append(a*output_img[i - 1, j + 1, 0] + b*warped_img[i - 1, j + 1, 0])
                listG.append(a*output_img[i - 1, j + 1, 1] + b*warped_img[i - 1, j + 1, 1])
                listR.append(a*output_img[i - 1, j + 1, 2] + b*warped_img[i - 1, j + 1, 2])
            #listB.append(warped_img[i - 1, j + 1, 0])
            #listG.append(warped_img[i - 1, j + 1, 1])
            #listR.append(warped_img[i - 1, j + 1, 2])
        if 0 <= i <= warped_img.shape[0]:
            if 0 <= j - 1 < warped_img.shape[1]:
                listB.append(a*output_img[i, j - 1, 0] + b*warped_img[i, j - 1, 0])
                listG.append(a*output_img[i, j - 1, 1] + b*warped_img[i, j - 1, 1])
                listR.append(a*output_img[i, j - 1, 2] + b*warped_img[i, j - 1, 2])
            #listB.append(warped_img[i, j - 1, 0])
            #listG.append(warped_img[i, j - 1, 1])
            #listR.append(warped_img[i, j - 1, 2])
            if 0 <= j < warped_img.shape[1]:
                listB.append(a*output_img[i, j, 0] + b*warped_img[i, j, 0])
                listG.append(a*output_img[i, j, 1] + b*warped_img[i, j, 1])
                listR.append(a*output_img[i, j, 2] + b*warped_img[i, j, 2])
                #listB.append(warped_img[i, j, 0])
                #listG.append(warped_img[i, j, 1])
                #listR.append(warped_img[i, j, 2])
            if 0 <= j + 1 < warped_img.shape[1]:
                listB.append(a*output_img[i, j + 1, 0] + b*warped_img[i, j + 1, 0])
                listG.append(a*output_img[i, j + 1, 1] + b*warped_img[i, j + 1, 1])
                listR.append(a*output_img[i, j + 1, 2] + b*warped_img[i, j + 1, 2])
            #listB.append(warped_img[i, j + 1, 0])
            #listG.append(warped_img[i, j + 1, 1])
            #listR.append(warped_img[i, j + 1, 2])
        if 0 <= i + 1 < warped_img.shape[0]:
            if 0 <= j - 1 < warped_img.shape[1]:
                listB.append(a*output_img[i + 1, j - 1, 0] + b*warped_img[i + 1, j - 1, 0])
                listG.append(a*output_img[i + 1, j - 1, 1] + b*warped_img[i + 1, j - 1, 1])
                listR.append(a*output_img[i + 1, j - 1, 2] + b*warped_img[i + 1, j - 1, 2])
            #listB.append(warped_img[i + 1, j - 1, 0])
            #listG.append(warped_img[i + 1, j - 1, 1])
            #listR.append(warped_img[i + 1, j - 1, 2])
            if 0 <= j < warped_img.shape[1]:
                listB.append(a*output_img[i + 1, j, 0] + b*warped_img[i + 1, j, 0])
                listG.append(a*output_img[i + 1, j, 1] + b*warped_img[i + 1, j, 1])
                listR.append(a*output_img[i + 1, j, 2] + b*warped_img[i + 1, j, 2])
            #listB.append(warped_img[i + 1, j, 0])
            #listG.append(warped_img[i + 1, j, 1])
            #listR.append(warped_img[i + 1, j, 2])
            if 0 <= j + 1 < warped_img.shape[1]:
                listB.append(a*output_img[i + 1, j + 1, 0] + b*warped_img[i + 1, j + 1, 0])
                listG.append(a*output_img[i + 1, j + 1, 1] + b*warped_img[i + 1, j + 1, 1])
                listR.append(a*output_img[i + 1, j + 1, 2] + b*warped_img[i + 1, j + 1, 2])
            #listB.append(warped_img[i + 1, j + 1, 0])
            #listG.append(warped_img[i + 1, j + 1, 1])
            #listR.append(warped_img[i + 1, j + 1, 2])
    listB.sort()
    listG.sort()
    listR.sort()
    idxB = (len(listB) + 1) // 2 - 1
    idxG = (len(listG) + 1) // 2 - 1
    idxR = (len(listR) + 1) // 2 - 1

    if idxB >= 0 and idxG >= 0 and idxR >= 0:
        B = listB[idxB]
        G = listG[idxG]
        R = listR[idxR]
        Rst = True

    return B, G, R, Rst

def checkDirection(frame, result, args):
    if is_cv2():  # opencv 버전 별
        sift = cv2.SIFT()
    elif is_cv3():
        sift = cv2.xfeatures2d.SIFT_create()
    elif is_cv4():
        sift = cv2.SIFT_create()
    else:
        raise RuntimeError("error! unknown version of python!")

    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 16}, {'checks': 50})

    computeFrame = frame.copy()
    computeResult = result.copy()
    computeFrame = imutils.resize(computeFrame, height=480)
    computeResult = imutils.resize(computeResult, height=480)

    features0 = sift.detectAndCompute(computeFrame, None)
    features1 = sift.detectAndCompute(computeResult, None)

    matches_src, matches_dst, n_matches = compute_matches(
        features0, features1, flann, knn=args.knn)

    H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
    direction = None
    x = computeResult.shape[1] // 2
    y = computeResult.shape[0] // 2
    x1 = (H[0][0] * x + H[0][1] * y + H[0][2]) / (H[2][0] * x + H[2][1] * y + 1)
    y1 = (H[1][0] * x + H[1][1] * y + H[1][2]) / (H[2][0] * x + H[2][1] * y + 1)
    moveX = x - int(x1)
    moveY = y - int(y1)
    if abs(moveX) >= abs(moveY):
        if moveX >= 0:
            direction = "left"
        else:
            direction = "right"
    else:
        if moveY >= 0:
            direction = "up"
        else:
            direction = "down"
    # print(str(x) + " " + str(y) + " " + str(int(x1)) + " " + str(int(y1)))
    # print("difference : " + str(x - int(x1)) + " " + str(y - int(y1)))
    #print(direction)

    return direction

def linearBlend(prev_img, warped_img, output_img, mask, direction):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # mulArea = cv2.findNonZero(thresh1)

    dicMax = {}
    dicMin = {}

    for cnt in contours:
        #cv2.drawContours(output_img, [cnt], -1, (0, 0, 255), 3)
        if direction is "right" or direction is "left":
            for xy in cnt[:, 0]:
                if xy[1] in dicMax:
                    if dicMax[xy[1]] < xy[0]:
                        dicMax[xy[1]] = xy[0]
                else:
                    dicMax[xy[1]] = xy[0]
                if xy[1] in dicMin:
                    if dicMin[xy[1]] > xy[0]:
                        dicMin[xy[1]] = xy[0]
                else:
                    dicMin[xy[1]] = xy[0]
        elif direction is "up" or direction is "down":
            for xy in cnt[:, 0]:
                if xy[0] in dicMax:
                    if dicMax[xy[0]] < xy[1]:
                        dicMax[xy[0]] = xy[1]
                else:
                    dicMax[xy[0]] = xy[1]
                if xy[0] in dicMin:
                    if dicMin[xy[0]] > xy[1]:
                        dicMin[xy[0]] = xy[1]
                else:
                    dicMin[xy[0]] = xy[1]

    for i in dicMin:
        #print(str(i) + " " + str(dicMin[i]) + " " + str(dicMax[i]))
        for x in range(dicMin[i], dicMax[i] + 1):
             ratio = (x - dicMin[i]) / (dicMax[i] - dicMin[i])

             if direction is "right":
                 output_img[i, x] = (1 - ratio) * prev_img[i, x] + ratio * warped_img[i, x]
             elif direction is "left":
                 output_img[i, x] = ratio * prev_img[i, x] + (1 - ratio) * warped_img[i, x]
             elif direction is "up":
                 output_img[x, i] = ratio * prev_img[x, i] + (1 - ratio) * warped_img[x, i]
             elif direction is "down":
                 output_img[x, i] = (1 - ratio) * prev_img[x, i] + ratio * warped_img[x, i]

    return output_img

def laplacian_blend(prev_img, warped_img, output_img, thresh):
    G = prev_img.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = warped_img.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize=size)
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        size = (gpB[i - 1].shape[1], gpB[i - 1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize=size)
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    cv2.imshow("SFddsfd", lpB[0])
    cv2.waitKey()
    LS = []
    xxx = 0
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = numpy.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1, 6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, LS[i])

    #output_img = output_img + ls_

    return ls_

def blend(frame1, frame2, mask1, mask2, type='multiband', blend_strength=20):
    # (x, y, w, h) = utils.helpers.returnRect(img2_warped)
    # (x1, y1, w1, h1) = utils.helpers.returnRect(resultT)
    (x, y, w, h) = (0, 0, frame1.shape[1], frame1.shape[0])
    (x1, y1, w1, h1) = (0, 0, frame2.shape[1], frame2.shape[0])
    corners = [(x, y), (x1, y1)]
    sizes = [(w, h), (w1, h1)]
    dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
    blend_width = numpy.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100

    if type == 'feather':
        blender = cv2.detail_FeatherBlender()
        blender.setSharpness(1. / blend_width)
    elif type == 'multiband':
        blender = cv2.detail_MultiBandBlender(try_gpu=1)
        blender.setNumBands((numpy.log(blend_width) / numpy.log(2.) - 1.).astype(numpy.int))

    blender.prepare(dst_sz)
    frame1_s = frame1.astype(numpy.int16)
    frame2_s = frame2.astype(numpy.int16)

    blender.feed(cv2.UMat(frame1_s), mask1, (0, 0))
    blender.feed(cv2.UMat(frame2_s), mask2, (0, 0))

    result = None
    result_mask = None
    result, result_mask = blender.blend(result, result_mask)
    result = result.astype(numpy.uint8)

    return result, result_mask

def rvContours(warped_img, n):
    img_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.medianBlur(thresh, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(warped_img, [cnt], 0, (0, 0, 0), n)
        cv2.drawContours(thresh, [cnt], 0, 0, n)

    warp_mask = cv2.bitwise_not(thresh)

    return warped_img, warp_mask

def returnRect(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.medianBlur(thresh, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        points = cv2.boundingRect(cnt)
    return points

def findPoint(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.medianBlur(thresh, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c0 in contours:
        leftmost = tuple(c0[c0[:, :, 0].argmin()][0])
        rightmost = tuple(c0[c0[:, :, 0].argmax()][0])
        topmost = tuple(c0[c0[:, :, 1].argmin()][0])
        bottommost = tuple(c0[c0[:, :, 1].argmax()][0])

    point = [leftmost, rightmost, topmost, bottommost]

    return point

def trapezoidalWarping(frame, direction, moved):
    ih, iw, _ = frame.shape
    black = numpy.zeros((ih, iw, 3), numpy.uint8)
    bh, bw, _ = black.shape
    pts_src = numpy.array([[0.0, 0.0], [float(iw), 0.0], [float(iw), float(ih)], [0.0, float(ih)]])
    movedX = 0
    #if direction is "left":
    #    moved = 3

    if moved > 0:
        movedX = 0.08
    elif moved < 0:
        movedX = -0.08
    elif moved == 0:
        movedX = 0.00

    #movedX = moved * (0.01 / 36) + (0.15 / 3)
    if movedX < 0:
        pts_dst = numpy.array([[float(bw * (-1 * movedX)), 0.0], [float(bw * (1 + movedX)), 0.0], [float(bw), float(bh)], [0.0, float(bh)]])
    if movedX > 0:
        pts_dst = numpy.array([[0.0, 0.0], [float(bw), 0.0], [float(bw * (1 - movedX)), float(bh)], [float(bw * movedX), float(bh)]])

    if movedX > 0 or movedX < 0:
        h, status = cv2.findHomography(pts_src, pts_dst)
        frame = cv2.warpPerspective(frame, h, (black.shape[1], black.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    #display("ab", frame)
    #cv2.waitKey(1)

    return frame

def eulerAnglesToRotationMatrix(theta):
    #Calculate rotation about x axis
    Rx = numpy.array([[1,0,0], [0,math.cos(theta[0]),-math.sin(theta[0])], [0,math.sin(theta[0]),math.cos(theta[0])]], dtype=numpy.float32)
    Ry = numpy.array([[math.cos(theta[1]),0,-math.sin(theta[1])], [0,1,0], [math.sin(theta[1]),0,math.cos(theta[1])]], dtype=numpy.float32)
    Rz = numpy.array([[math.cos(theta[2]),-math.sin(theta[2]),0], [math.sin(theta[2]),math.cos(theta[2]),0], [0,0,1]], dtype=numpy.float32)
    #Combined rotation matrix
    R = Rz * Ry * Rx

    return R