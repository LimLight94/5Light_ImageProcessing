import logging
# Standard Modules
import cv2
import copy
import numpy as np
import imutils
from ImageProcessing_py import utils
import time

# Custom Modules


logger = logging.getLogger('main')


def combine_images(stitched_img, newFrame, h_matrix, resultT):
    logger.debug('combining images... ')
    h0, w0 = stitched_img.shape[:2]

    max_w = w0
    max_h = h0

    # ##case1!!!
    img2_warped = cv2.warpPerspective(newFrame, h_matrix, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    tm = cv2.TickMeter()
    tm.reset()
    tm.start()
    t1 = time.time()

    img2_warped, warp_mask = utils.helpers.rvContours(img2_warped)
    output_mask = utils.helpers.ifExist(resultT)
    mask = cv2.bitwise_or(output_mask, warp_mask)
    output_img = cv2.bitwise_and(resultT, resultT, mask=mask)
    output_img1 = output_img + img2_warped
    warp_mask1 = cv2.bitwise_not(warp_mask)
    output_mask1 = cv2.bitwise_not(output_mask)
    mask1 = cv2.bitwise_not(mask)
    if np.count_nonzero(mask1) != 0:
        blend_strength = 20

        #(x, y, w, h) = utils.helpers.returnRect(img2_warped)
        #(x1, y1, w1, h1) = utils.helpers.returnRect(resultT)
        (x, y, w, h) = (0, 0, img2_warped.shape[1], img2_warped.shape[0])
        (x1, y1, w1, h1) = (0, 0, resultT.shape[1], resultT.shape[0])
        corners = [(x, y), (x1, y1)]
        sizes = [(w, h), (w1, h1)]
        dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100

        #blender = cv2.detail_FeatherBlender()
        #blender.setSharpness(1. / blend_width)
        #blender.setSharpness(30)
        #print("sharpness : " + str(1. / blend_width)) #0.021
        blender = cv2.detail_MultiBandBlender(try_gpu=1)
        blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
        #blender.setNumBands(1)
        #print("num : " + str((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))) #6
        blender.prepare(dst_sz)
        img2_warped_s = img2_warped.astype(np.int16)
        resultT_s = resultT.astype(np.int16)
        blender.feed(cv2.UMat(resultT_s), output_mask1, (0, 0))
        blender.feed(cv2.UMat(img2_warped_s), warp_mask1, (0,0))

        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        result = result.astype(np.uint8)

        tm.stop()
        ms = tm.getTimeMilli()  # 밀리 초 단위 시간을 받아옴

        #print('blend time:', (time.time() - t1) * 1000)
        #print('blend Elapsed time: {}ms.'.format(tm.getTimeMilli()))
        utils.helpers.display("output", output_img1)
        #cv2.imshow("fdsdfsdf", output_img1)
        cv2.waitKey(1)

    return output_img1, tm.getTimeMilli()