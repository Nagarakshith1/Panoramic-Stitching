'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

import cv2
import numpy as np
import annoy
from annoy import *


def feat_match(descs1, descs2):
	# Your Code Here

	print('desc shape', descs1.shape)

	annoy_index = AnnoyIndex(descs1.shape[0], metric="euclidean")

	for i in range(descs1.shape[1]):
		annoy_index.add_item(i, descs2[:, i])

	annoy_index.build(50)

	matched_index = np.array([0 for i in range(descs1.shape[1])])

	for i in range(descs1.shape[1]):
		id, distances = annoy_index.get_nns_by_vector(descs1[:, i], n=2, include_distances=True)
		if distances[0] < 0.7 * distances[1]:
			matched_index[i] = id[0]
		else:
			matched_index[i] = -1

	return matched_index
