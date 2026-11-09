from libc.math cimport sin, cos, sqrt, M_PI
import numpy as np
import scipy.sparse as sp

__all__ = [
	'build_hamiltonian'
]

cdef int check_state(unsigned int s, int N, int K):
	"""
	Check the state `s` for periodicity and return the period if valid.

	Parameters:
	s (unsigned int): The state to check.
	N (int): The number of bits in the state.
	k (int): The momentum value.

	Returns:
	int: The period of the state if valid, otherwise -1.
	"""
	cdef unsigned int t = s
	cdef int i

	for i in range(1, N + 1):
		t = ((t >> 1) | ((t & 1) << (N - 1)))

		if t < s:
			return -1

		if t == s:
			if K % (N // i) != 0:
				return -1
			return i

	return -1

cdef inline int popcount(unsigned int s):
	"""
	Calculate the population count (number of set bits) in an unsigned integer.

	Args:
	s (unsigned int): The input unsigned integer.

	Returns:
	int: The count of set bits in the input integer.
	"""
	cdef int count = 0
	while s:
		count += s & 1
		s = s >> 1
	return count

cdef inline int parity(unsigned int s):
	"""
	Find the parity of the population of the state `s`.

	Parameters:
	s (unsigned int): The state to find the parity for.

	Returns:
	int: 0 if population is even, 1 if population is odd.
	"""
	return popcount(s) & 1

cdef tuple generate_basis(int N, dict sector):
	"""
	Generate the momentum and/or magnetization and/or fermion parity basis states and their periodicities.

	Parameters:
	N (int): The number of bits in the state.
	sector (dict): A dictionary containing 'M' and/or 'K' and/or 'P' keys representing the magnetization value and/or momentum value and/or fermion parity value.

	Returns:
	tuple: A tuple containing a list of basis states and their periodicities.
	"""
	cdef list basis = []
	cdef list periodicities = []

	cdef int s, m, R, p

	for s in range(1 << N):

		if 'M' in sector:
			m = popcount(s)
			if m != sector['M']:
				continue

		if 'P' in sector:
			p = parity(s)
			if p != sector['P']:
				continue

		if 'K' in sector:
			R = check_state(s, N, sector['K'])
			if R <= 0:
				continue

			basis.append(s)
			periodicities.append(R)

		else:
			basis.append(s)

	return basis, periodicities

cdef tuple find_representative(unsigned int s, int N):
	"""
	Find the lexicographically smallest representative of the state `s`.

	Parameters:
	s (unsigned int): The state to find the representative for.
	N (int): The number of bits in the state.

	Returns:
	tuple: A tuple containing the representative state and its period.
	"""
	cdef int r = s
	cdef int t = s
	cdef int i, l = N

	for i in range(1, N):
		t = ((t >> 1) | ((t & 1) << (N - 1)))
		if t < r:
			r = t
			l = i

	return r, l

cdef int find_state(unsigned int s, list basis):
	"""
	Find the index of the state `s` in the sorted list `basis`.

	Parameters:
	s (unsigned int): The state to find.
	basis (list): The sorted list of basis states.

	Returns:
	int: The index of the state in the basis, or -1 if not found.
	"""
	cdef int bmin = 0, bmax = len(basis) - 1, b

	while bmin <= bmax:
		b = bmin + (bmax - bmin) // 2
		if s < basis[b]:
			bmax = b - 1
		elif s > basis[b]:
			bmin = b + 1
		else:
			return b

	return -1

cdef inline double fast_sqrt(int a, int b):
	"""
	Calculate the square root of the ratio of two integers.

	Parameters:
	a (int): The numerator of the ratio.
	b (int): The denominator of the ratio. Must be non-zero to avoid division by zero.

	Returns:
	double: The square root of the ratio of `a` to `b`.
	"""
	return sqrt(<double> a / <double> b)

cdef inline complex fast_exp(double angle):
	"""
	Calculate the complex exponential of a given angle.

	Parameters:
	angle (double): The angle in radians for which to compute the complex exponential.

	Returns:
	complex: The complex exponential of the given angle.
	"""
	return cos(angle) + 1j * sin(angle)

cdef dict apply_hamiltonian(dict model, dict params, unsigned int state):
	"""
	Apply the Hamiltonian to a given state and return the resulting states.

	Parameters:
	model (dict): The model parameters including the model type.
	params (dict): The parameters including 'theta', 'lam', and 'N'.
	state (unsigned int): The initial state.

	Returns:
	dict: A dictionary of new states with their corresponding weights.
	"""
	cdef dict new_states = {}
	cdef double cos_t = cos(params['theta'])
	cdef double sin_t = sin(params['theta'])
	cdef double weight_1 = cos(params['lam'])
	cdef double weight_2 = sin(params['lam'])
	cdef int j, flipped_xx, flipped_yy, sign, z_sign
	cdef int N = params['N']
	
	cdef bint PBC = True if model['BCs'] == "PBC" else False
	
	for j in range(N - (0 if PBC else 1)):
		flipped_xx = state ^ (1 << j) ^ (1 << ((j + 1) % N))
		flipped_yy = flipped_xx
		sign = -1 if ((state >> j) & 1) == ((state >> ((j + 1) % N)) & 1) else 1

		if model['name'] == "A":
			new_states[flipped_xx] = new_states.get(flipped_xx, 0) + weight_1 * cos_t
			new_states[flipped_yy] = new_states.get(flipped_yy, 0) + weight_1 * sin_t * sign
		elif model['name'] == "B":
			new_states[flipped_xx] = new_states.get(flipped_xx, 0) + weight_1 * cos_t
			new_states[flipped_yy] = new_states.get(flipped_yy, 0) + weight_1 * sin_t * sign
		elif model['name'] == "C":
			new_states[flipped_xx] = new_states.get(flipped_xx, 0) + weight_1 * cos_t
			new_states[flipped_yy] = new_states.get(flipped_yy, 0) + weight_2 * sin_t * sign

	for j in range(N - (0 if PBC else 3)):
		z_sign = 1 - 2 * (((state >> j) & 1) ^ ((state >> ((j + 3) % N)) & 1))
		flipped_xx = state ^ (1 << ((j + 1) % N)) ^ (1 << ((j + 2) % N))
		flipped_yy = flipped_xx
		sign = -1 if ((state >> ((j + 1) % N)) & 1) == ((state >> ((j + 2) % N)) & 1) else 1

		if model['name'] == "A":
			new_states[flipped_xx] = new_states.get(flipped_xx, 0) + weight_2 * cos_t * z_sign
			new_states[flipped_yy] = new_states.get(flipped_yy, 0) + weight_2 * sin_t * sign * z_sign
		elif model['name'] == "B":
			new_states[flipped_xx] = new_states.get(flipped_xx, 0) + weight_2 * sin_t * z_sign
			new_states[flipped_yy] = new_states.get(flipped_yy, 0) + weight_2 * cos_t * sign * z_sign
		elif model['name'] == "C":
			new_states[flipped_xx] = new_states.get(flipped_xx, 0) + weight_1 * sin_t * z_sign
			new_states[flipped_yy] = new_states.get(flipped_yy, 0) + weight_2 * cos_t * sign * z_sign

	return new_states

cpdef tuple build_hamiltonian(dict model, dict params, dict sector):
	"""
	Construct the Hamiltonian in a given symmetry sector (momentum and/or magnetization).

	Parameters:
	model (dict): Model parameters including type (e.g., 'name') and boundary conditions (e.g., 'BCs').
	params (dict): Hamiltonian parameters including 'theta', 'lam', and number of sites 'N'.
	sector (dict): Sector definition, can include keys:
				   - 'M' for magnetization
				   - 'K' for momentum
				   - 'P' for fermion parity

	Returns:
	tuple: A 2-tuple:
		   - H: Sparse matrix of the Hamiltonian in the sector
		   - basis: List of representative bitstrings in the sector
	"""
	cdef int N = params['N']
	cdef list basis, periodicities

	basis, periodicities = generate_basis(N, sector)

	cdef int dim = len(basis)
	cdef object H = sp.lil_matrix((dim, dim), dtype=np.complex128)

	cdef int i, j, translation
	cdef object new_state, representative
	cdef dict new_states
	cdef double complex coeff, phase
	cdef double norm, theta

	if 'K' in sector:
		theta = -2.0 * M_PI * sector['K'] / N

		for i in range(dim):
			new_states = apply_hamiltonian(model, params, basis[i])

			for new_state, coeff in new_states.items():
			
				representative, translation = find_representative(new_state, N)
				j = find_state(representative, basis)

				if j >= 0:
					phase = fast_exp(theta * translation)
					norm = fast_sqrt(periodicities[i], periodicities[j])
					H[i, j] += coeff * phase * norm

	else:
		for i in range(dim):
			new_states = apply_hamiltonian(model, params, basis[i])

			for new_state, coeff in new_states.items():

				j = find_state(new_state, basis)
				if j >= 0:
					H[i, j] += coeff

	return H.tocsr(), basis
