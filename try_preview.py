# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:17:43 2020

@author: Evgeniy
"""
import cv2
import pytesseract
import numpy as np


def get_grayscale(image): return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # noise removal
def remove_noise(image): return cv2.medianBlur(image,5) #thresholding
def thresholding(image): return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #dilation
def dilate(image): 
    kernel = np.ones((5,5),np.uint8) 
    return cv2.dilate(image, kernel, iterations = 1) #erosion
def erode(image): 
    kernel = np.ones((5,5),np.uint8) 
    return cv2.erode(image, kernel, iterations = 1) #opening - erosion followed by dilation
def opening(image): 
    kernel = np.ones((5,5),np.uint8) 
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) #canny edge detection
def canny(image): return cv2.Canny(image, 100, 200) #skew correction
def deskew(image): 
    coords = np.column_stack(np.where(image > 0)) 
    angle = cv2.minAreaRect(coords)[-1] 
    if angle < -45: 
        angle = -(90 + angle) 
    else: 
        angle = -angle 
        (h, w) = image.shape[:2] 
        center = (w // 2, h // 2) 
        M = cv2.getRotationMatrix2D(center, angle, 1.0) 
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
    return rotated #template matching
def match_template(image, template): return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


# Подключение фото
img = cv2.imread('example_crop.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

scale_percent = 90 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
gray = cv2.cvtColor(img, cv2.COLOR)

thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)
# resize image
resized = cv2.resize(canny, dim, interpolation = cv2.INTER_AREA)
# Будет выведен весь текст с картинки

# Делаем нечто более крутое!!!

#data = pytesseract.image_to_data(img, config=config)

# Перебираем данные про текстовые надписи

# Отображаем фото
cv2.imshow('Result', resized)
#cv2.waitKey(0)