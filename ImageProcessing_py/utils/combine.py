import logging
# Standard Modules
import cv2
from ImageProcessing_py import utils
# Custom Modules

logger = logging.getLogger('main')


def combine_images(stitched_img, newFrame, h_matrix, resultT):
    logger.debug('combining images... ')

    #warp image
    h0, w0 = stitched_img.shape[:2]
    img2_warped = cv2.warpPerspective(newFrame, h_matrix, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    #merge image
    img2_warped, warp_mask = utils.helpers.rvContours(img2_warped, 3)
    prev_mask = utils.helpers.ifExist(resultT)
    output_mask = cv2.bitwise_or(prev_mask, warp_mask)
    output_mask_and = cv2.bitwise_and(prev_mask, warp_mask)
    output_img_sub = cv2.bitwise_and(resultT, resultT, mask=output_mask)
    output_img = output_img_sub + img2_warped

    #find warped image contours
    img2_warped_gray = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img2_warped_gray, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.medianBlur(thresh, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #blending(multiband, feather)
    '''warp_mask_not = cv2.bitwise_not(warp_mask)
    prev_mask_not = cv2.bitwise_not(prev_mask)
    output_mask_not = cv2.bitwise_not(output_mask_and)
    if np.count_nonzero(output_mask_not) != 0:
        result, result_mask = utils.helpers.blend(img2_warped, resultT, warp_mask_not, prev_mask_not)'''

    return output_img, contours[0]