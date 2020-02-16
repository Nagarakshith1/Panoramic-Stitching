'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''
from est_homography import est_homography
import numpy as np
import math

def ransac_est_homography(x1, y1, x2, y2, thresh):
	# Your Code Here

	no_of_features = len(x1)
	max_no_of_inliers = 0
	final_inlier_indices = []
	H_list =[]
	final_inlier_indices_list=[]
	update_flag = 0
	for i in range(4000):
		random_indices = np.random.choice(no_of_features, 4, replace=False)

		#print(x1[random_indices].reshape(1,4)[0])
		H = est_homography(x1[random_indices].reshape(1,4)[0], y1[random_indices].reshape(1,4)[0], x2[random_indices].reshape(1,4)[0], y2[random_indices].reshape(1,4)[0])
		H = H/H[2,2]

		inlier_ind = np.zeros(no_of_features, dtype=int)
		inlier_ind[random_indices[0]] = 1
		inlier_ind[random_indices[1]] = 1
		inlier_ind[random_indices[2]] = 1
		inlier_ind[random_indices[3]] = 1

		no_of_inliers = 0
		for j in range(no_of_features):
			if j not in random_indices:
				point_u = np.array([int(x1[j]), int(y1[j]), 1]).reshape(3,1)
				point_v = np.matmul(H, point_u)
				point_v = point_v / point_v[2]
				#print(point_v)
				#print('diff', abs(point_v[0] - x2[j]), point_v[0], x2[j])
				#print('diff', abs(point_v[1] - y2[j]), point_v[1], y2[j])

				#if abs(point_v[0] - x2[j])<=thresh and abs(point_v[1] - y2[j])<=thresh:
				if math.sqrt((point_v[0] - x2[j])**2 + (point_v[1] - y2[j])**2)<= thresh:
					no_of_inliers += 1
					inlier_ind[j] = 1

		if no_of_inliers > max_no_of_inliers:
			max_no_of_inliers = no_of_inliers
			final_inlier_indices = np.copy(inlier_ind)
			update_flag = 1

		if update_flag==1:
			H_list=[]
			final_inlier_indices_list=[]
			update_flag=0

		if no_of_inliers == max_no_of_inliers:
			H_list.append(H)
			final_inlier_indices_list.append(final_inlier_indices)

	list_size = len(H_list)
	min_error = np.inf
	for i in range(list_size):
		H = H_list[i]
		final_inlier_indices = final_inlier_indices_list[i]

		x_points = x1[final_inlier_indices==1].reshape(1,-1)
		y_points = y1[final_inlier_indices==1].reshape(1,-1)



		stack = np.vstack((x_points,y_points,np.ones((x_points.shape))))

		homo_stack = np.matmul(H,stack)
		homo_stack = homo_stack/homo_stack[2]

		x2_homo_points = homo_stack[0]
		y2_homo_points = homo_stack[1]

		x2_points = x2[final_inlier_indices==1]
		y2_points = y2[final_inlier_indices == 1]

		error = np.sum((x2_points-x2_homo_points)**2+(y2_points-y2_homo_points)**2)

		if(error<min_error):
			min_error = error
			H_list_index = i

	# H = H_list[H_list_index]
	final_inlier_indices=final_inlier_indices_list[H_list_index]

	# print('final_inlier_indices', final_inlier_indices)

	final_homography = est_homography(x1[final_inlier_indices==1].reshape(1,max_no_of_inliers+4)[0], y1[final_inlier_indices==1].reshape(1,max_no_of_inliers+4)[0], x2[final_inlier_indices==1].reshape(1,max_no_of_inliers+4)[0], y2[final_inlier_indices==1].reshape(1,max_no_of_inliers+4)[0])
	final_homography = final_homography/final_homography[2,2]
	print('final_homography',final_homography)
	return final_homography, final_inlier_indices