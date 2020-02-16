'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

import numpy as np
import scipy
from scipy import signal
import cv2

def GaussianPDF_1D(mu, sigma, length):
  # create an array
  half_len = length / 2

  if np.remainder(length, 2) == 0:
    ax = np.arange(-half_len, half_len, 1)
  else:
    ax = np.arange(-half_len, half_len + 1, 1)

  ax = ax.reshape([-1, ax.size])
  denominator = sigma * np.sqrt(2 * np.pi)
  nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

  return nominator / denominator

def GaussianPDF_2D(mu, sigma, row, col):
  # create row vector as 1D Gaussian pdf
  g_row = GaussianPDF_1D(mu, sigma, row)
  # create column vector as 1D Gaussian pdf
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()

  return scipy.signal.convolve2d(g_row, g_col, 'full')

def feat_desc(img, x, y):
  # Your Code Here
  print('feat_desc')
  G = GaussianPDF_2D(0, 1, 4, 4)
  dx, dy = np.gradient(G, axis=(1, 0))

  # print(dx, dy)

  Ix = scipy.signal.convolve(img, dx, 'same')
  Iy = scipy.signal.convolve(img, dy, 'same')
  Im = np.sqrt(Ix * Ix + Iy * Iy)

  Im = np.pad(Im, ((20, 20), (20, 20)), 'constant')

  #Ori = np.arctan2(Iy, Ix)
  #Ori = np.pad(Ori, ((20, 20), (20, 20)), 'constant')

  no_of_features =len(x)

  descs = [[] for i in range(no_of_features)]
  #descs = np.zeros((512,no_of_features))

  for i in range(no_of_features):
    x1 = int(x[i] + 20)
    y1 = int(y[i] + 20)
    #print(x1-19 , x1+21)
    path_of_40_40 = Im[y1-19 : y1+21, x1-19 : x1+21]
    # print('patch')

    #cv2.imwrite('gg.jpg', path_of_40_40)
    vsplit_patches = np.hstack(np.vsplit(path_of_40_40, 8))
    final_5_5_patches = np.array(np.hsplit(vsplit_patches, 64))

    maximum_horizontally = np.amax(final_5_5_patches, axis=2)
    vector_before_normalization = np.amax(maximum_horizontally, axis=1)
    descs[i] = (vector_before_normalization - np.mean(vector_before_normalization)) / np.std(vector_before_normalization)
  return np.array(descs).T
  # truth_mat = np.zeros(8,64)
  # for i in range(no_of_features):
  #   x1 = int(x[i] + 20)
  #   y1 = int(y[i] + 20)
  #   patch_of_40_40 = Ori[y1-19 : y1+21, x1-19 : x1+21]
  #   k = np.hsplit(patch_of_40_40,8)
  #   l = np.vstack((k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]))
  #   truth = np.logical_and(l>=0,l<45)
  #   truth_mat[0] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   truth = np.logical_and(l>=45,l<90)
  #   truth_mat[1] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   truth = np.logical_and(l>=90,l<135)
  #   truth_mat[2] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   truth = np.logical_and(l>=135,l<180)
  #   truth_mat[3] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   truth = np.logical_and(l>=180,l<225)
  #   truth_mat[4] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   truth = np.logical_and(l>=225,l<270)
  #   truth_mat[5] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   truth = np.logical_and(l>=270,l<315)
  #   truth_mat[6] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   truth = np.logical_and(l>=315,l<360)
  #   truth_mat[7] = np.sum(np.sum(truth,axis=1).reshape(-1,5),axis=1)
  #   value = truth_mat.reshape(-1,1)[:,0]
  #   descs[:,i] = (value-np.mean(value))/np.std(value)
  #  return descs