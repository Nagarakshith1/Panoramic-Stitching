'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
import numpy as np
from scipy import sparse
import time
import matplotlib.pyplot as plt


def anms(cimg, max_pts):
	# Your Code Here

	cimg[cimg < 0.01 * np.max(cimg)] = 0

	no_of_non_zeros = len(cimg[cimg != 0])
	sparse = np.array([[0.0, 0.0, 0.0] for i in range(no_of_non_zeros)])

	x = 0
	for i in range(cimg.shape[0]):
		for j in range(cimg.shape[1]):
			if cimg[i, j] != 0:
				sparse[x] = np.array([i, j, cimg[i, j]])
				x += 1


	sparse.view('i8,i8,i8').sort(order=['f2'], axis=0)
	mininum_radii = np.array([np.inf for i in range(no_of_non_zeros)])

	for i in range(no_of_non_zeros):

		value = sparse[i, 2]
		find_radius_from_points = sparse[sparse[:, 2] > 0.9 * value]

		if find_radius_from_points.shape[0] != 0:
			mininum_radii[i] = np.min(np.sqrt(np.sum((find_radius_from_points[:, :2] - sparse[i, :2]) ** 2, axis=1)))

	sorted_radii_index = np.flip(np.argsort(mininum_radii))
	x = sparse[sorted_radii_index, 1]
	y = sparse[sorted_radii_index, 0]

	return x[0:max_pts].reshape(-1, 1), y[0:max_pts].reshape(-1, 1)
