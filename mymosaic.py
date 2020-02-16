'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

from interp2 import interp2

from cumMinVer import cumMinEngVer
from rmVerSeam import rmVerSeam
from genEngMap import genEngMap

import numpy as np
from PIL import Image


def stitch(img1, img2, H, x_shift, y_shift, middle_image, value):
	xx, yy = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
	all_index = np.vstack((xx.flatten(), yy.flatten()))
	all_index = np.vstack((all_index, np.ones(all_index.shape[1])))

	new_homography_matrix = np.matmul([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]], H)

	final_points_indices = np.matmul(new_homography_matrix, all_index)

	final_points_indices = final_points_indices / final_points_indices[2]
	final_points_indices = final_points_indices.astype(int)
	all_index = all_index.astype(int)
	intermediate_image = np.copy(img2)

	x_min_new = np.min(final_points_indices[0])
	y_min_new = np.min(final_points_indices[1])

	x_max_new = np.max(final_points_indices[0])
	y_max_new = np.max(final_points_indices[1])

	xx_new, yy_new = np.meshgrid(np.arange(int(x_min_new), int(x_max_new)), np.arange(int(y_min_new), int(y_max_new)))

	stitched_index = np.vstack((xx_new.flatten(), yy_new.flatten()))
	stitched_index = np.vstack((stitched_index, np.ones(stitched_index.shape[1])))

	H_inv = np.linalg.inv(new_homography_matrix)
	new_image_points = np.matmul(H_inv, stitched_index)

	new_image_points = new_image_points / new_image_points[2]

	pixels_in_bounds = np.logical_and(np.all(new_image_points >= 0, axis=0), new_image_points[0] < img1.shape[1])
	pixels_in_bounds = np.logical_and(pixels_in_bounds, new_image_points[1] < img1.shape[0])

	stitched_index = stitched_index[:, pixels_in_bounds].astype(int)
	new_image_points = new_image_points[:, pixels_in_bounds]

	red_channel = img1[:, :, 0]
	green_channel = img1[:, :, 1]
	blue_channel = img1[:, :, 2]

	red_pixel_values_from_img1 = interp2(red_channel, new_image_points[0], new_image_points[1])
	green_pixel_values_from_img1 = interp2(green_channel, new_image_points[0], new_image_points[1])
	blue_pixel_values_from_img1 = interp2(blue_channel, new_image_points[0], new_image_points[1])

	intermediate_image[stitched_index[1], stitched_index[0], 0] = red_pixel_values_from_img1
	intermediate_image[stitched_index[1], stitched_index[0], 1] = green_pixel_values_from_img1
	intermediate_image[stitched_index[1], stitched_index[0], 2] = blue_pixel_values_from_img1

	seam = seam_carving(x_shift, y_shift, new_homography_matrix, img1, img2, middle_image, value)

	if value == 1:
		for min_point in seam:
			y_seam = min_point[0]
			x_seam = min_point[1]

			intermediate_image[y_seam, x_seam: middle_image.shape[1] + int(x_shift), :] = middle_image[
																						  y_seam - int(y_shift),
																						  x_seam - int(x_shift):, :]

	else:
		homo_of_corner_point_1 = np.matmul(new_homography_matrix, np.array([0, 0, 1]).reshape(3, 1))
		homo_of_corner_point_1 = homo_of_corner_point_1 / homo_of_corner_point_1[2]

		homo_of_corner_point_2 = np.matmul(new_homography_matrix, np.array([0, img1.shape[0] - 1, 1]).reshape(3, 1))
		homo_of_corner_point_2 = homo_of_corner_point_2 / homo_of_corner_point_2[2]

		if homo_of_corner_point_1[0] < homo_of_corner_point_2[0]:
			min_x_co_ordinate = int(homo_of_corner_point_1[0])
		else:
			min_x_co_ordinate = int(homo_of_corner_point_2[0])

		for min_point in seam:
			y_seam = min_point[0]
			x_seam = min_point[1]

			intermediate_image[y_seam, int(min_x_co_ordinate):x_seam, :] = middle_image[y_seam - int(y_shift),
																		   int(min_x_co_ordinate) - int(
																			   x_shift):x_seam - int(x_shift), :]

	return intermediate_image


def do_the_actual_seam_carving(bounding_box):
	e = genEngMap(bounding_box)
	Mx, Tbx = cumMinEngVer(e)
	seam_to_be_removed = rmVerSeam(bounding_box, Mx, Tbx)
	return seam_to_be_removed


def seam_carving(x_shift, y_shift, final_homography, img1, img2, middle_image, value):
	if value == 1:
		homo_of_corner_point_1 = np.matmul(final_homography, np.array([img1.shape[1] - 1, 0, 1]).reshape(3, 1))
		homo_of_corner_point_1 = homo_of_corner_point_1 / homo_of_corner_point_1[2]

		homo_of_corner_point_2 = np.matmul(final_homography,
										   np.array([img1.shape[1] - 1, img1.shape[0] - 1, 1]).reshape(3, 1))
		homo_of_corner_point_2 = homo_of_corner_point_2 / homo_of_corner_point_2[2]

		if homo_of_corner_point_1[0] > homo_of_corner_point_2[0]:
			max_x_co_ordinate = int(homo_of_corner_point_1[0])
		else:
			max_x_co_ordinate = int(homo_of_corner_point_2[0])

		bounding_box = middle_image[:, 0:max_x_co_ordinate - int(x_shift), :]

		seam = do_the_actual_seam_carving(bounding_box)

		seam[:, 0] = seam[:, 0] + int(y_shift)
		seam[:, 1] = seam[:, 1] + int(x_shift)

		return seam

	else:
		homo_of_corner_point_1 = np.matmul(final_homography, np.array([0, 0, 1]).reshape(3, 1))
		homo_of_corner_point_1 = homo_of_corner_point_1 / homo_of_corner_point_1[2]

		homo_of_corner_point_2 = np.matmul(final_homography, np.array([0, img1.shape[0] - 1, 1]).reshape(3, 1))
		homo_of_corner_point_2 = homo_of_corner_point_2 / homo_of_corner_point_2[2]

		if homo_of_corner_point_1[0] < homo_of_corner_point_2[0]:
			max_x_co_ordinate = int(homo_of_corner_point_1[0])
		else:
			max_x_co_ordinate = int(homo_of_corner_point_2[0])

		bounding_box = middle_image[:, max_x_co_ordinate - int(x_shift):, :]

		seam = do_the_actual_seam_carving(bounding_box)

		x_list = seam[:, 1]
		y_list = seam[:, 0]

		bounding_box[y_list, x_list, :] = [255, 0, 0]

		seam[:, 0] = seam[:, 0] + int(y_shift)
		seam[:, 1] = seam[:, 1] + max_x_co_ordinate

		return seam


def mymosaic(img_left, img_middle, img_right, H12, H32):
	h_pad = int(img_middle.shape[0] / 1.5)
	w_pad = int(img_middle.shape[1] * 1.2)

	img_middle_pad = np.pad(img_middle, ((h_pad, h_pad), (w_pad, w_pad), (0, 0)), 'constant')

	stitched_img1 = stitch(img_left, img_middle_pad, H12, w_pad, h_pad, img_middle, value=1)
	iii = Image.fromarray(stitched_img1, "RGB")
	# iii.show()
	# iii.save('final_left_middle_stitched_building.jpg')

	stitched_img2 = stitch(img_right, stitched_img1, H32, w_pad, h_pad, img_middle, value=2)
	iii = Image.fromarray(stitched_img2, "RGB")
	iii.show()
	# iii.save('final_stitched_building.jpg')

	return stitched_img2
