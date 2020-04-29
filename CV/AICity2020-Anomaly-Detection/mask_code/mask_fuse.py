import cv2
import sys
video_name = sys.argv[1]

mask_bg = cv2.imread("data/mask_diff/" + str(video_name).zfill(3) + ".jpg",0)
kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
mask_dilate = cv2.erode(mask_bg, kernel_e)
mask_dilate = cv2.dilate(mask_dilate, kernel)

mask_det = cv2.imread("data/mask_track/mask_" + str(video_name) + ".png",0)

mask_dilate[mask_dilate<127] = 0
mask_dilate[mask_dilate>=127] = 1

mask_det[mask_det<127] = 0
mask_det[mask_det>=127] = 1

mask_res = (mask_dilate&mask_det)

cv2.imwrite("./data/mask_" + str(video_name).zfill(3) + ".jpg", mask_res*255)
