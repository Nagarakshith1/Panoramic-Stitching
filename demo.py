from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from mymosaic import mymosaic

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

from PIL import Image


def rgb2gray(I_rgb):
	r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
	I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return I_gray


def open_image(filename):
	img = Image.open(filename).convert('RGB')
	img = np.array(img)
	gray = rgb2gray(img)
	gray = gray.astype('float32')

	return img, gray


def plot_image_matching(x1, y1, x2, y2, bx1, by1, bx2, by2, img1, img2):
	f = plt.figure(figsize=(30, 30))
	ax1 = f.add_subplot(121)
	ax2 = f.add_subplot(122)

	ax1.imshow(img1)
	ax1.axis('off')
	ax2.imshow(img2)
	ax2.axis('off')

	ax1.plot(x1, y1, 'ro', markersize=3)
	ax2.plot(x2, y2, 'ro', markersize=3)

	ax1.plot(bx1, by1, 'bo', markersize=3)
	ax2.plot(bx2, by2, 'bo', markersize=3)

	for i in range(len(x1)):
		con = ConnectionPatch(xyA=(x2[i], y2[i]), xyB=(x1[i], y1[i]), coordsA="data", coordsB="data", axesA=ax2,
							  axesB=ax1, color="red")
		ax2.add_artist(con)

	for i in range(len(bx1)):
		con = ConnectionPatch(xyA=(bx2[i], by2[i]), xyB=(bx1[i], by1[i]), coordsA="data", coordsB="data", axesA=ax2,
							  axesB=ax1, color="blue")
		ax2.add_artist(con)

	#plt.show()


def plot_corner_harris(img, filename):
	iii = Image.fromarray(img, 'RGB')
	#iii.show()
	#iii.save(filename + '.jpg')

def plot_anms(img, x, y):
	f = plt.figure(figsize=(30, 30))
	ax1 = f.add_subplot(121)
	ax1.imshow(img)
	ax1.plot(x, y, 'ro', markersize=2)

	#plt.show()

def main():
	max_pts = 1000

	left_image_filename = 'building1.JPG'
	middle_image_filename = 'building2.JPG'
	right_image_filename = 'building3.JPG'

	img_left, gray_left = open_image(left_image_filename)
	img_middle, gray_middle = open_image(middle_image_filename)
	img_right, gray_right = open_image(right_image_filename)

	corners_left = corner_detector(gray_left)
	corners_middle = corner_detector(gray_middle)
	corners_right = corner_detector(gray_right)

	plotting_img_left = np.copy(img_left)
	plotting_img_middle = np.copy(img_middle)
	plotting_img_right = np.copy(img_right)

	print('co', corners_left, corners_left.shape)
	plotting_img_left[corners_left >= 0.01 * np.max(corners_left)] = [255, 0, 0]
	plotting_img_middle[corners_middle >= 0.01 * np.max(corners_left)] = [255, 0, 0]
	plotting_img_right[corners_right >= 0.01 * np.max(corners_left)] = [255, 0, 0]

	plot_corner_harris(plotting_img_left, 'corner_harris_left_2')
	plot_corner_harris(plotting_img_middle, 'corner_harris_middle_2')
	plot_corner_harris(plotting_img_right, 'corner_harris_right_2')

	x_left, y_left = anms(corners_left, max_pts)
	x_middle, y_middle = anms(corners_middle, max_pts)
	x_right, y_right = anms(corners_right, max_pts)

	plot_anms(img_left, x_left, y_left)
	plot_anms(img_middle, x_middle, y_middle)
	plot_anms(img_right, x_right, y_right)

	desc_left = feat_desc(gray_left, x_left, y_left)
	desc_middle = feat_desc(gray_middle, x_middle, y_middle)
	desc_right = feat_desc(gray_right, x_right, y_right)

	matched_index_left_middle = feat_match(desc_left, desc_middle)
	x1 = x_left[matched_index_left_middle != -1]
	y1 = y_left[matched_index_left_middle != -1]
	x2 = x_middle[matched_index_left_middle[matched_index_left_middle != -1]]
	y2 = y_middle[matched_index_left_middle[matched_index_left_middle != -1]]

	thresh = 0.5
	H12, inlier_ind = ransac_est_homography(x1, y1, x2, y2, thresh)

	bx1 = x1[inlier_ind == 0]
	by1 = y1[inlier_ind == 0]
	bx2 = x2[inlier_ind == 0]
	by2 = y2[inlier_ind == 0]

	x1 = x1[inlier_ind != 0]
	y1 = y1[inlier_ind != 0]
	x2 = x2[inlier_ind != 0]
	y2 = y2[inlier_ind != 0]

	plot_image_matching(x1, y1, x2, y2, bx1, by1, bx2, by2, img_left, img_middle)

	matched_index_right_middle = feat_match(desc_right, desc_middle)

	x1 = x_right[matched_index_right_middle != -1]
	y1 = y_right[matched_index_right_middle != -1]
	x2 = x_middle[matched_index_right_middle[matched_index_right_middle != -1]]
	y2 = y_middle[matched_index_right_middle[matched_index_right_middle != -1]]

	thresh = 1
	H32, inlier_ind = ransac_est_homography(x1, y1, x2, y2, thresh)

	bx1 = x1[inlier_ind == 0]
	by1 = y1[inlier_ind == 0]
	bx2 = x2[inlier_ind == 0]
	by2 = y2[inlier_ind == 0]

	x1 = x1[inlier_ind != 0]
	y1 = y1[inlier_ind != 0]
	x2 = x2[inlier_ind != 0]
	y2 = y2[inlier_ind != 0]

	plot_image_matching(x1, y1, x2, y2, bx1, by1, bx2, by2, img_right, img_middle)

	mymosaic(img_left, img_middle, img_right, H12, H32)


main()
