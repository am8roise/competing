import numpy as np

__all__ = [
	'SUIrrep'
]

class SUIrrep:
	"""
	Class representing an SU(N) irrep.
	"""

	def __init__(self, N: int, dim: int = None):
		"""
		Initialize an SU(N) irrep.

		Args:
			N (int): The SU(N) group.
			dim (int, optional): The dimension of the irrep. Default is N.
		"""
		self.N = N
		self.dim = dim if dim else N

		self.generators = self.construct_generators()

	def construct_generators(self):
		"""
		Construct generators based on the SU(N) group.

		Returns:
			tuple: Tuple of generators for the SU(N) group.
		"""
		if self.N == 2:
			return self.su2_generators()
		else:
			raise NotImplementedError(f"Generators for SU({self.N}) not implemented yet.")

	def su2_generators(self):
		"""
		Construct generators for SU(2) group.

		Returns:
			tuple: Tuple of generators for the SU(2) group.
		"""
		
		if self.dim < 2 or self.dim > 11:
			raise NotImplementedError(f"Generators for SU({self.N}) with dimension {self.dim} not implemented yet. Allowed dimensions are 2, ..., 11.")

		if self.dim == 2:
			t1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
			t2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
			t3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
		else:
			s = (self.dim - 1) / 2.0
			vals = np.arange(s, -s - 1, -1, dtype=np.float64)

			p = np.zeros((self.dim, self.dim), dtype=np.complex128)

			for i in range(1, self.dim):
				m = vals[i]
				coeff = np.sqrt(s * (s + 1) - m * (m + 1))
				p[i - 1, i] = coeff

			m = p.conj().T

			t1 = 0.5 * (p + m)
			t2 = -0.5j * (p - m)
			t3 = np.diag(vals).astype(np.complex128)

		return (t1, t2, t3)

	def __repr__(self):
		"""
		Return a string representation of the object for debugging purposes.
		
		Returns:
			str: A string containing the class name, value of N, and dimension of the object.
		"""
		return f"<'SUIrrep', N={self.N}, dim={self.dim}>"
