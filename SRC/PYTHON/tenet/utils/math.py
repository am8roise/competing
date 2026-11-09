import numpy as np

__all__ = [
	'magnitude',
	'r2',
	'find_nearest',
	'PI'
]

PI = np.pi

def magnitude(x):
	"""
	Calculate the order of magnitude of a given number.

	Parameters:
	x (float): The number for which to calculate the magnitude.

	Returns:
	int: The order of magnitude of the number.
	"""
	return int(np.floor(np.log10(abs(x))))

def r2(y, fit):
	"""
	Calculate the coefficient of determination (R-squared) for a given set of observations and predictions.

	Parameters:
	y (array-like): The observed data points.
	fit (array-like): The predicted data points from a model.

	Returns:
	float: The R-squared value indicating the goodness of fit.
	"""
	ss_res = np.sum((y-fit)**2)
	ss_tot = np.sum((y-np.mean(y))**2)

	return (1 - ss_res/ss_tot)

def find_nearest(array, value):
	"""
	Find the index of the nearest value in an array to a given target value.

	Parameters:
	array (array-like): The array to search through.
	value (float): The target value to find the nearest to.

	Returns:
	int: The index of the nearest value in the array.
	"""
	array = np.asarray(array)
	idx = np.abs(array - value).argmin()

	return idx
