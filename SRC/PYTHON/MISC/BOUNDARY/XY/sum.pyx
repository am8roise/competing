from libc.math cimport sin, cos, M_PI
import numpy as np
cimport numpy as np

cpdef double compute_sum(int n):
	cdef int half = n // 2
	cdef int i1, i2, i3
	cdef double s = 0.0
	cdef double sign
	cdef int parity

	if n < 4:
		return 0.0

	cdef np.ndarray[np.float64_t, ndim=1] sins = np.empty(half, dtype=np.float64)
	cdef np.ndarray[np.float64_t, ndim=1] coss = np.empty(half, dtype=np.float64)
	cdef np.ndarray[np.float64_t, ndim=1] sin2s = np.empty(half, dtype=np.float64)

	cdef double[:] sins_view = sins
	cdef double[:] coss_view = coss
	cdef double[:] sin2s_view = sin2s

	cdef double pi_over = M_PI / (n - 1)

	for i1 in range(1, half):
		sins_view[i1] = sin(pi_over * i1)
		coss_view[i1] = cos(pi_over * i1)
		sin2s_view[i1] = sin(2.0 * pi_over * i1)

	for i1 in range(1, half):
		for i2 in range(1, half):
			for i3 in range(1, half):
				parity = (i1 + i2 + i3 + 1) & 1
				sign = 1.0 if parity == 0 else -1.0

				s += (
					sign *
					sins_view[i1] *
					sin2s_view[i2] *
					(sin2s_view[i3] * sin2s_view[i3]) *
					(sins_view[i1] * sin2s_view[i2] - sin2s_view[i1] * sins_view[i2])
					/ (2.0 * coss_view[i1] + 2.0 * coss_view[i2] + 2.0 * coss_view[i3])
				)

	return s
