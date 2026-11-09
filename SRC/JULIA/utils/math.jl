using PyCall

py"""
def magnitude(x):
	import numpy as np

	return int(np.floor(np.log10(abs(x))))
"""
py"""
def r2(y, fit):
	import numpy as np

	ss_res = np.sum((y-fit)**2)
	ss_tot = np.sum((y-np.mean(y))**2)

	return (1 - ss_res/ss_tot)
"""

py"""
def find_nearest(array, value):
	import numpy as np

	array = np.asarray(array)
	idx = np.abs(array - value).argmin()

	return idx
"""