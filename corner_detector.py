'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''

import cv2
import numpy as np


def corner_detector(img):
  # Your Code Here
  dst_before_dilate = cv2.cornerHarris(img, 2, 3, 0.04)
  #dst_after_dilate = cv2.dilate(dst_before_dilate, None)

  return dst_before_dilate