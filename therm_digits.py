# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:17:43 2020

@author: Evgeniy
"""
# import the necessary packages
import numpy as np
import cv2
import imutils
from skimage import exposure
import pytesseract
import PIL
from PIL import ImageOps
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import argparse
import random as rng


def cnvt_edged_image(img_arr, should_save=False):
  # ratio = img_arr.shape[0] / 300.0
  image = imutils.resize(img_arr,height=300)
  gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),11, 17, 17)
  edged_image = cv2.Canny(gray_image, 30, 200)

  if should_save:
    cv2.imwrite('cntr_ocr.jpg',edged_image)

  return edged_image

'''image passed in must be ran through the cnv_edge_image first'''
def find_display_contour(edge_img_arr):
  display_contour = None
  edge_copy = edge_img_arr.copy()
  contours,hierarchy = cv2.findContours(edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

  for cntr in top_cntrs:
    peri = cv2.arcLength(cntr,True)
    approx = cv2.approxPolyDP(cntr, 0.02 * peri, True)

    if len(approx) == 4:
      display_contour = approx
      break

  return display_contour

def crop_display(image_arr):
  edge_image = cnvt_edged_image(image_arr)
  display_contour = find_display_contour(edge_image)
  cntr_pts = display_contour.reshape(4,2)
  return cntr_pts


def normalize_contrs(img,cntr_pts):
  ratio = img.shape[0] / 300.0
  norm_pts = np.zeros((4,2), dtype="float32")

  s = cntr_pts.sum(axis=1)
  norm_pts[0] = cntr_pts[np.argmin(s)]
  norm_pts[2] = cntr_pts[np.argmax(s)]

  d = np.diff(cntr_pts,axis=1)
  norm_pts[1] = cntr_pts[np.argmin(d)]
  norm_pts[3] = cntr_pts[np.argmax(d)]

  norm_pts *= ratio

  (top_left, top_right, bottom_right, bottom_left) = norm_pts

  width1 = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
  width2 = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
  height1 = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
  height2 = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

  max_width = max(int(width1), int(width2))
  max_height = max(int(height1), int(height2))

  dst = np.array([[0,0], [max_width -1, 0],[max_width -1, max_height -1],[0, max_height-1]], dtype="float32")
  persp_matrix = cv2.getPerspectiveTransform(norm_pts,dst)
  return cv2.warpPerspective(img,persp_matrix,(max_width,max_height))

def process_image(orig_image_arr):
  ratio = orig_image_arr.shape[0] / 300.0

  display_image_arr = normalize_contrs(orig_image_arr,crop_display(orig_image_arr))
  #display image is now segmented.
  gry_disp_arr = cv2.cvtColor(display_image_arr, cv2.COLOR_BGR2GRAY)
  gry_disp_arr = exposure.rescale_intensity(gry_disp_arr, out_range= (0,255))

  #thresholding
  ret, thresh = cv2.threshold(gry_disp_arr,127,255,cv2.THRESH_BINARY)
  return thresh

def ocr_image(orig_image_arr):
  otsu_thresh_image = PIL.Image.fromarray(process_image(orig_image_arr))
  otsu_thresh_image = otsu_thresh_image.convert('L')
  #otsu_thresh_image = ImageOps.invert(otsu_thresh_image)
  #otsu_thresh_image.save('result.jpg')
  scale_percent = 49 # percent of original size
  width = int(otsu_thresh_image.width * scale_percent / 100)
  height = int(otsu_thresh_image.height * scale_percent / 100)
  dim = (width, height)
  
  otsu_thresh_image = otsu_thresh_image.resize(dim)
  return otsu_thresh_image

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 30 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

# load the example image
img = cv2.imread("ex1.jpg")
gray=ocr_image(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('first_stage',gray)
#cv2.imshow("Input", gray)
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

cv2.imshow('thresh',gray)
cnts,hierarchy = cv2.findContours(gray.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

LENGTH = len(cnts)
status = np.zeros((LENGTH,1))

for i,cnt1 in enumerate(cnts):
    x = i    
    if i != LENGTH-1:
        for j,cnt2 in enumerate(cnts[i+1:]):
            x = x+1
            dist = find_if_close(cnt1,cnt2)
            if dist == True:
                val = min(status[i],status[x])
                status[x] = status[i] = val
            else:
                if status[x]==status[i]:
                    status[x] = i+1
unified = []
maximum = int(status.max())+1
for i in range(maximum):
    pos = np.where(status==i)[0]
    if pos.size != 0:
        cont = np.vstack(cnts[i] for i in pos)
        hull = cv2.convexHull(cont)
        unified.append(hull)
unified_areas=[]
for i in unified:
    area = cv2.contourArea(i)
    unified_areas.append(area)
digits=[]
for i in unified:
    area = cv2.contourArea(i)
    if area>20000:
        digits.append(i)      
rect0_x,rect0_y,rect0_w,rect0_h=cv2.boundingRect(digits[0])
rect1_x,rect1_y,rect1_w,rect1_h=cv2.boundingRect(digits[1])
rect2_x,rect2_y,rect2_w,rect2_h=cv2.boundingRect(digits[2])
if rect0_x==min([rect0_x,rect1_x,rect2_x]):
    cv2.rectangle(img,(rect0_x,rect0_y),(rect0_x+rect0_w,rect0_y+rect0_h),color=(0,0,255))
    cv2.imshow("Box", img)
    digit0 = img[rect0_y:rect0_y+rect0_h,rect0_x:rect0_x+rect0_w]
    digit0=cv2.resize(digit0,(28,28))
    cv2.imshow("Digit0", digit0)
    if rect1_x==min([rect1_x,rect2_x]):
        cv2.rectangle(img,(rect1_x,rect1_y),(rect1_x+rect1_w,rect1_y+rect1_h),color=(0,0,255))
        cv2.imshow("Box", img)
        digit1 = img[rect1_y:rect1_y+rect1_h,rect1_x:rect1_x+rect1_w]
        digit1=cv2.resize(digit1,(28,28))
        cv2.imshow("Digit1", digit1)
        cv2.rectangle(img,(rect2_x,rect2_y),(rect2_x+rect2_w,rect2_y+rect2_h),color=(0,0,255))
        cv2.imshow("Box", img)
        digit2 = img[rect2_y:rect2_y+rect2_h,rect2_x:rect2_x+rect2_w]
        digit2=cv2.resize(digit2,(28,28))
        cv2.imshow("Digit2", digit2)
        
if rect1_x==min([rect0_x,rect1_x,rect2_x]):
    cv2.rectangle(img,(rect1_x,rect1_y),(rect1_x+rect1_w,rect1_y+rect0_h),color=(0,0,255))
    cv2.imshow("Box", img)
    digit0 = img[rect1_y:rect1_y+rect1_h,rect1_x:rect1_x+rect1_w]
    digit0=cv2.resize(digit0,(28,28))
    cv2.imshow("Digit0", digit0)
    if rect0_x==min([rect0_x,rect2_x]):
        cv2.rectangle(img,(rect0_x,rect0_y),(rect0_x+rect0_w,rect0_y+rect0_h),color=(0,0,255))
        cv2.imshow("Box", img)
        digit1 = img[rect0_y:rect0_y+rect0_h,rect0_x:rect0_x+rect0_w]
        digit1=cv2.resize(digit1,(28,28))
        cv2.imshow("Digit1", digit1)
        cv2.rectangle(img,(rect2_x,rect2_y),(rect2_x+rect2_w,rect2_y+rect2_h),color=(0,0,255))
        cv2.imshow("Box", img)
        digit2 = img[rect2_y:rect2_y+rect2_h,rect2_x:rect2_x+rect2_w]
        digit2=cv2.resize(digit2,(28,28))
        cv2.imshow("Digit2", digit2)
if rect2_x==min([rect0_x,rect1_x,rect2_x]):
    cv2.rectangle(img,(rect2_x,rect2_y),(rect2_x+rect2_w,rect2_y+rect2_h),color=(0,0,255))
    cv2.imshow("Box", img)
    digit0 = img[rect2_y:rect2_y+rect2_h,rect2_x:rect2_x+rect2_w]
    digit0=cv2.resize(digit0,(28,28))
    cv2.imshow("Digit0", digit0)
    if rect0_x==min([rect0_x,rect1_x]):
        cv2.rectangle(img,(rect0_x,rect0_y),(rect0_x+rect0_w,rect0_y+rect0_h),color=(0,0,255))
        cv2.imshow("Box", img)
        digit1 = img[rect0_y:rect0_y+rect0_h,rect0_x:rect0_x+rect0_w]
        digit1=cv2.resize(digit1,(28,28))
        cv2.imshow("Digit1", digit1)
        cv2.rectangle(img,(rect1_x,rect1_y),(rect1_x+rect1_w,rect1_y+rect1_h),color=(0,0,255))
        cv2.imshow("Box", img)
        digit2 = img[rect1_y:rect1_y+rect1_h,rect1_x:rect1_x+rect1_w]
        digit2=cv2.resize(digit2,(28,28))
        cv2.imshow("Digit2", digit2)
        
cv2.drawContours(img, cnts, 10, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)
cv2.imshow("Input", img)







