from libc.math cimport sin, cos, M_PI

cdef inline double eps(int n, int N):
	return 2.0 * cos(3.141592653589793 * n / (N - 1))

cdef inline double s(int k, int n, int N):
	return sin(k * 3.141592653589793 * n / (N - 1))

cpdef double compute_sum(
	int N,
	int[:] low, int[:] high,
	int[:] sin_k,
	int[:] phase_pow,
	int[:] denom_sign
):
	cdef int n1, n2, n3, n4
	cdef double res = 0.0
	cdef double denom, num, phase

	for n1 in range(low[0], high[0]+1):
		for n2 in range(low[1], high[1]+1):
			for n3 in range(low[2], high[2]+1):
				for n4 in range(low[3], high[3]+1):

					phase = (-1.0) ** (
						phase_pow[0]*n1 +
						phase_pow[1]*n2 +
						phase_pow[2]*n3 +
						phase_pow[3]*n4
					)

					num = 1.0
					if sin_k[0]: num *= s(sin_k[0], n1, N)
					if sin_k[1]: num *= s(sin_k[1], n2, N)
					if sin_k[2]: num *= s(sin_k[2], n3, N)
					if sin_k[3]: num *= s(sin_k[3], n4, N)
					if sin_k[4]: num *= s(sin_k[4], n1, N)
					if sin_k[5]: num *= s(sin_k[5], n2, N)
					if sin_k[6]: num *= s(sin_k[6], n3, N)
					if sin_k[7]: num *= s(sin_k[7], n4, N)

					denom = (
						denom_sign[0] * eps(n1, N) +
						denom_sign[1] * eps(n2, N) +
						denom_sign[2] * eps(n3, N) +
						denom_sign[3] * eps(n4, N)
					)

					if denom != 0.0:
						res += phase * num / denom

	return res
