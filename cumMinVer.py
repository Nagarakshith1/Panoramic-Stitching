import numpy as np


def cumMinEngVer(e):
	# Your Code Here
	Mx = np.zeros(e.shape)
	Tbx = np.zeros(e.shape)
	Mx[0] = e[0]

	no_of_columns = e.shape[1] - 1

	mini = np.zeros(e.shape[1])

	for i in range(1, e.shape[0]):
		left_shifted = np.hstack((Mx[i - 1, 1:], np.inf))

		right_shifted = np.hstack((np.inf, Mx[i - 1, 0:no_of_columns]))

		np.minimum(left_shifted, right_shifted, mini)
		np.minimum(mini, Mx[i - 1], mini)

		Mx[i] = e[i] + mini

		stack_of_left_right = np.array([right_shifted, Mx[i - 1], left_shifted])
		Tbx[i] = np.argmin(stack_of_left_right, axis=0) - 1

	return Mx, Tbx
