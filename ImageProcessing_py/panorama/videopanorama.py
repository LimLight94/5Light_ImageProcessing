import matplotlib.pyplot as plt
import argparse
import logging #디버그할 때 로그찍어주는 모듈
import cv2  #opencv 모듈
import numpy as np
import sys

from ImageProcessing_py import utils

if __name__ == '__main__':  # 플러그
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video_path', type=str, help="paths to one or more images or image directories")
    parser.add_argument('-b', '--debug', dest='debug', action='store_true', help='enable debug logging')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='disable all logging')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help="display result")
    parser.add_argument('-s', '--save', dest='save', action='store_true', help="save result to file")
    parser.add_argument("--save_path", dest='save_path', default="stitched.png", type=str, help="path to save result")
    parser.add_argument('-k', '--knn', dest='knn', default=2, type=int, help="Knn cluster value")
    parser.add_argument('-l', '--lowe', dest='lowe', default=0.7, type=float, help='acceptable distance between points')
    parser.add_argument('-m', '--min', dest='min_correspondence', default=10, type=int, help='min correspondences')
    args = parser.parse_args()

    if args.debug: # 예외처리
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    logging.info("beginning sequential matching")

    if utils.helpers.is_cv2():     #opencv 버전 별
        sift = cv2.SIFT()
    elif utils.helpers.is_cv3():
        sift = cv2.xfeatures2d.SIFT_create()
    elif utils.helpers.is_cv4():
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        raise RuntimeError("error! unknown version of python!")

    result = None
    result_gry = None

    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

    cap = cv2.VideoCapture(args.video_path)
    i=0
    while True:

        ret, frame = cap.read()
        if i%10 != 0:
            i= i+1
            continue
        i = i + 1


        if not ret: #제대로 못읽었을 경우
            break
        else:       #제대로 읽었을 경우 그레이로 바꾸기
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if result is None:
            result = frame
        else:
            features0 = sift.detectAndCompute(result_gry, None)
            features1 = sift.detectAndCompute(frame_gray, None)

            matches_src, matches_dst, n_matches = utils.compute_matches(
                features0, features1, flann, knn=args.knn)

            if n_matches < args.min_correspondence:
                logger.error("error! too few correspondences")
                continue

            H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
            result = utils.combine_images(frame, result, H)

            if args.display and not args.quiet:
                utils.helpers.display('result', result)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        result_gry = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    logger.info('{0} is completed'.format(args.video_path))
    cap.release()
    cv2.destroyAllWindows()
    if args.save:
        utils.helpers.save_image(args.video_path, result)


# 이미지가 불러져왔는지 확인
image = cv2.imread("C:/All/repos/5Light_ImageProcessing/res/panorama/test.png", cv2.IMREAD_ANYCOLOR)
if image is None:
    print('Imager load failed!') # 이미지가 없으면 출력
    sys.exit()
cv2.imshow("panorama image", image)

# 마스크 구하기(픽셀값 0,255로 나누기)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1])
thresh = cv2.medianBlur(thresh, 5)

plt.figure(figsize=(20, 20))
plt.imshow(thresh, cmap='gray')


# 테두리 지우기
stitched_copy = image.copy()
thresh_copy = thresh.copy()

while np.sum(thresh_copy) > 0:
    thresh_copy = thresh_copy[1:-1, 1:-1]
    stitched_copy = stitched_copy[1:-1, 1:-1]

plt.figure(figsize=(20, 20))
plt.imshow(cv2.cvtColor(stitched_copy, cv2.COLOR_BGR2RGB))
cv2.imshow("clean image", stitched_copy)
cv2.imwrite("C:/All/repos/5Light_ImageProcessing/res/panorama/result.png",stitched_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()