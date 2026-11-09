import numpy as np
from ncon import ncon

from tenet.utils.decorators import *

__all__ = [
	'MPO',
	'iMPO',
	'Binder',
]

class MPO:

	def __init__(self, N, p, model, params):
		"""
		Initialize the class instance with provided parameters.

		Args:
		N (int): Number of sites.
		p (object): Irreducible representation.
		model (dict): Model information.
		params (dict): Additional parameters.

		Returns:
		None
		"""

		self.N = N
		self.p = p
		self.model = model
		self.params = params

		self.model["PBCs"] = model.get('PBCs', False)

		self.d = p.dim
		self.T = p.generators

		self._I = np.eye(self.d)

		if p.N == 2:
			self._X, self._Y, self._Z = [self.T[i] for i in range(3)]

		if model["name"] == "A":
			self._initParams_A()
			self._initLocal_A()
			self._initMPO_A(model["PBCs"])

		elif model["name"] == "B":
			self._initParams_B()
			self._initLocal_B()
			self._initMPO_B(model["PBCs"])

		elif model["name"] == "C":
			self._initParams_C()
			self._initLocal_C()
			self._initMPO_C(model["PBCs"])

	def _initParams_A(self):
		"""
		Initialize parameters specific to model A.

		Args:
		None

		Returns:
		None
		"""

		self._lam = self.params['lam']
		self._theta = self.params['theta']

	def _initParams_B(self):
		"""
		Initialize parameters specific to model B.

		Args:
		None

		Returns:
		None
		"""

		self._lam = self.params['lam']
		self._theta = self.params['theta']

	def _initParams_C(self):
		"""
		Initialize parameters specific to model B.

		Args:
		None

		Returns:
		None
		"""

		self._lam = self.params['lam']
		self._theta = self.params['theta']

	def _initLocal_A(self):
		"""
		Initialize local Hamiltonian specific to model A.

		Args:
		None

		Returns:
		None
		"""

		pass

	def _initLocal_B(self):
		"""
		Initialize local Hamiltonian specific to model B.

		Args:
		None

		Returns:
		None
		"""

		pass

	def _initLocal_C(self):
		"""
		Initialize local Hamiltonian specific to model B.

		Args:
		None

		Returns:
		None
		"""

		pass

	def _initMPO_A(self, PBCs):
		"""
		Initialize the matrix product operator (MPO) A based on the given periodic boundary conditions (PBCs).

		Args:
		self: The object instance.
		PBCs (bool): A flag indicating whether to apply periodic boundary conditions.

		Returns:
		None
		"""
		if PBCs:

			H = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			H[0, :, 0, :] = self._I
			H[1, :, 0, :] = self._X
			H[2, :, 0, :] = self._Y
			H[3, :, 0, :] = self._Z

			H[4, :, 1, :] = self._I
			H[5, :, 2, :] = self._I
			H[6, :, 3, :] = self._I
			H[7, :, 6, :] = self._X
			H[8, :, 6, :] = self._Y
			H[9, :, 7, :] = self._I
			H[10, :, 8, :] = self._I
			H[11, :, 9, :] = self._X
			H[12, :, 10, :] = self._Y
			H[13, :, 11, :] = self._I
			H[14, :, 12, :] = self._I

			H[38, :, 4, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			H[38, :, 5, :] = np.cos(self._lam)*np.sin(self._theta)*self._Y
			H[38, :, 13, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z
			H[38, :, 14, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z

			H[38, :, 38, :] = self._I

			self.mpo = [H.copy() for i in range(self.N)]

			# -----

			HH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HH[38, :, 1, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			HH[38, :, 2, :] = np.cos(self._lam)*np.sin(self._theta)*self._Y

			self.mpo[0] += HH.copy()
			self.mpo[-2] += HH.copy()

			# -----

			HHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHH[15, :, 1, :] = self._Z
			HHH[16, :, 2, :] = self._Z
			HHH[17, :, 15, :] = self._X
			HHH[18, :, 16, :] = self._Y
			HHH[19, :, 17, :] = self._I
			HHH[20, :, 18, :] = self._I

			HHH[38, :, 19, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z
			HHH[38, :, 20, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z

			self.mpo[-5] += HHH.copy()
			self.mpo[-4] += HHH.copy()
			self.mpo[-3] += HHH.copy()
			self.mpo[-2] += HHH.copy()

			# -----

			HHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHH[21, :, 1, :] = self._X
			HHHH[22, :, 2, :] = self._Y
			HHHH[23, :, 21, :] = self._Z
			HHHH[24, :, 22, :] = self._Z

			HHHH[38, :, 23, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z
			HHHH[38, :, 24, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z

			self.mpo[-4] += HHHH.copy()
			self.mpo[-3] += HHHH.copy()
			self.mpo[-2] += HHHH.copy()

			# -----

			HHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHH[25, :, 3, :] = self._X
			HHHHH[26, :, 3, :] = self._Y
			HHHHH[27, :, 25, :] = self._I
			HHHHH[28, :, 26, :] = self._I
			HHHHH[29, :, 27, :] = self._X
			HHHHH[30, :, 28, :] = self._Y
			HHHHH[31, :, 29, :] = self._I
			HHHHH[32, :, 30, :] = self._I

			HHHHH[38, :, 31, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z
			HHHHH[38, :, 32, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z

			self.mpo[-6] += HHHHH.copy()
			self.mpo[-5] += HHHHH.copy()
			self.mpo[-4] += HHHHH.copy()
			self.mpo[-3] += HHHHH.copy()
			self.mpo[-2] += HHHHH.copy()

			# -----

			HHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHH[38, :, 11, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z
			HHHHHH[38, :, 12, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z

			self.mpo[0] += HHHHHH.copy()

			# -----

			HHHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHHH[33, :, 3, :] = self._Z
			HHHHHHH[34, :, 33, :] = self._X
			HHHHHHH[35, :, 33, :] = self._Y

			HHHHHHH[38, :, 34, :] = np.sin(self._lam)*np.cos(self._theta)*self._X
			HHHHHHH[38, :, 35, :] = np.sin(self._lam)*np.sin(self._theta)*self._Y

			self.mpo[0] += HHHHHHH.copy()
			self.mpo[1] += HHHHHHH.copy()
			self.mpo[2] += HHHHHHH.copy()

			# -----

			HHHHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHHHH[36, :, 7, :] = self._Z
			HHHHHHHH[37, :, 8, :] = self._Z

			HHHHHHHH[38, :, 36, :] = np.sin(self._lam)*np.cos(self._theta)*self._X
			HHHHHHHH[38, :, 37, :] = np.sin(self._lam)*np.sin(self._theta)*self._Y

			self.mpo[0] += HHHHHHHH.copy()
			self.mpo[1] += HHHHHHHH.copy()

		else:

			H = np.zeros([9, self.d, 9, self.d], dtype=np.complex128)

			H[0, :, 0, :] = self._I
			H[1, :, 0, :] = self._X
			H[2, :, 0, :] = self._Y
			H[3, :, 0, :] = self._Z

			H[4, :, 3, :] = self._X
			H[5, :, 3, :] = self._Y
			H[6, :, 4, :] = self._X
			H[7, :, 5, :] = self._Y

			H[8, :, 1, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			H[8, :, 2, :] = np.cos(self._lam)*np.sin(self._theta)*self._Y
			H[8, :, 6, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z
			H[8, :, 7, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z

			H[8, :, 8, :] = self._I

			self.mpo = [H.copy() for i in range(self.N)]

	def _initMPO_B(self, PBCs):
		"""
		Initialize the matrix product operator (MPO) A based on the given periodic boundary conditions (PBCs).

		Args:
		self: The object instance.
		PBCs (bool): A flag indicating whether to apply periodic boundary conditions.

		Returns:
		None
		"""
		if PBCs:

			H = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			H[0, :, 0, :] = self._I
			H[1, :, 0, :] = self._X
			H[2, :, 0, :] = self._Y
			H[3, :, 0, :] = self._Z

			H[4, :, 1, :] = self._I
			H[5, :, 2, :] = self._I
			H[6, :, 3, :] = self._I
			H[7, :, 6, :] = self._X
			H[8, :, 6, :] = self._Y
			H[9, :, 7, :] = self._I
			H[10, :, 8, :] = self._I
			H[11, :, 9, :] = self._X
			H[12, :, 10, :] = self._Y
			H[13, :, 11, :] = self._I
			H[14, :, 12, :] = self._I

			H[38, :, 4, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			H[38, :, 5, :] = np.cos(self._lam)*np.sin(self._theta)*self._Y
			H[38, :, 13, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z
			H[38, :, 14, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			H[38, :, 38, :] = self._I

			self.mpo = [H.copy() for i in range(self.N)]

			# -----

			HH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HH[38, :, 1, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			HH[38, :, 2, :] = np.cos(self._lam)*np.sin(self._theta)*self._Y

			self.mpo[0] += HH.copy()
			self.mpo[-2] += HH.copy()

			# -----

			HHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHH[15, :, 1, :] = self._Z
			HHH[16, :, 2, :] = self._Z
			HHH[17, :, 15, :] = self._X
			HHH[18, :, 16, :] = self._Y
			HHH[19, :, 17, :] = self._I
			HHH[20, :, 18, :] = self._I

			HHH[38, :, 19, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z
			HHH[38, :, 20, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[-5] += HHH.copy()
			self.mpo[-4] += HHH.copy()
			self.mpo[-3] += HHH.copy()
			self.mpo[-2] += HHH.copy()

			# -----

			HHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHH[21, :, 1, :] = self._X
			HHHH[22, :, 2, :] = self._Y
			HHHH[23, :, 21, :] = self._Z
			HHHH[24, :, 22, :] = self._Z

			HHHH[38, :, 23, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z
			HHHH[38, :, 24, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[-4] += HHHH.copy()
			self.mpo[-3] += HHHH.copy()
			self.mpo[-2] += HHHH.copy()

			# -----

			HHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHH[25, :, 3, :] = self._X
			HHHHH[26, :, 3, :] = self._Y
			HHHHH[27, :, 25, :] = self._I
			HHHHH[28, :, 26, :] = self._I
			HHHHH[29, :, 27, :] = self._X
			HHHHH[30, :, 28, :] = self._Y
			HHHHH[31, :, 29, :] = self._I
			HHHHH[32, :, 30, :] = self._I

			HHHHH[38, :, 31, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z
			HHHHH[38, :, 32, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[-6] += HHHHH.copy()
			self.mpo[-5] += HHHHH.copy()
			self.mpo[-4] += HHHHH.copy()
			self.mpo[-3] += HHHHH.copy()
			self.mpo[-2] += HHHHH.copy()

			# -----

			HHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHH[38, :, 11, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z
			HHHHHH[38, :, 12, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[0] += HHHHHH.copy()

			# -----

			HHHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHHH[33, :, 3, :] = self._Z
			HHHHHHH[34, :, 33, :] = self._X
			HHHHHHH[35, :, 33, :] = self._Y

			HHHHHHH[38, :, 34, :] = np.sin(self._lam)*np.sin(self._theta)*self._X
			HHHHHHH[38, :, 35, :] = np.sin(self._lam)*np.cos(self._theta)*self._Y

			self.mpo[0] += HHHHHHH.copy()
			self.mpo[1] += HHHHHHH.copy()
			self.mpo[2] += HHHHHHH.copy()

			# -----

			HHHHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHHHH[36, :, 7, :] = self._Z
			HHHHHHHH[37, :, 8, :] = self._Z

			HHHHHHHH[38, :, 36, :] = np.sin(self._lam)*np.sin(self._theta)*self._X
			HHHHHHHH[38, :, 37, :] = np.sin(self._lam)*np.cos(self._theta)*self._Y

			self.mpo[0] += HHHHHHHH.copy()
			self.mpo[1] += HHHHHHHH.copy()

		else:

			H = np.zeros([9, self.d, 9, self.d], dtype=np.complex128)

			H[0, :, 0, :] = self._I
			H[1, :, 0, :] = self._X
			H[2, :, 0, :] = self._Y
			H[3, :, 0, :] = self._Z

			H[4, :, 3, :] = self._X
			H[5, :, 3, :] = self._Y
			H[6, :, 4, :] = self._X
			H[7, :, 5, :] = self._Y

			H[8, :, 1, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			H[8, :, 2, :] = np.cos(self._lam)*np.sin(self._theta)*self._Y
			H[8, :, 6, :] = np.sin(self._lam)*np.sin(self._theta)*self._Z
			H[8, :, 7, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			H[8, :, 8, :] = self._I

			self.mpo = [H.copy() for i in range(self.N)]
		
	def _initMPO_C(self, PBCs):
		"""
		Initialize the matrix product operator (MPO) A based on the given periodic boundary conditions (PBCs).

		Args:
		self: The object instance.
		PBCs (bool): A flag indicating whether to apply periodic boundary conditions.

		Returns:
		None
		"""
		if PBCs:

			H = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			H[0, :, 0, :] = self._I
			H[1, :, 0, :] = self._X
			H[2, :, 0, :] = self._Y
			H[3, :, 0, :] = self._Z

			H[4, :, 1, :] = self._I
			H[5, :, 2, :] = self._I
			H[6, :, 3, :] = self._I
			H[7, :, 6, :] = self._X
			H[8, :, 6, :] = self._Y
			H[9, :, 7, :] = self._I
			H[10, :, 8, :] = self._I
			H[11, :, 9, :] = self._X
			H[12, :, 10, :] = self._Y
			H[13, :, 11, :] = self._I
			H[14, :, 12, :] = self._I

			H[38, :, 4, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			H[38, :, 5, :] = np.sin(self._lam)*np.sin(self._theta)*self._Y
			H[38, :, 13, :] = np.cos(self._lam)*np.sin(self._theta)*self._Z
			H[38, :, 14, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			H[38, :, 38, :] = self._I

			self.mpo = [H.copy() for i in range(self.N)]

			# -----

			HH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HH[38, :, 1, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			HH[38, :, 2, :] = np.sin(self._lam)*np.sin(self._theta)*self._Y

			self.mpo[0] += HH.copy()
			self.mpo[-2] += HH.copy()

			# -----

			HHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHH[15, :, 1, :] = self._Z
			HHH[16, :, 2, :] = self._Z
			HHH[17, :, 15, :] = self._X
			HHH[18, :, 16, :] = self._Y
			HHH[19, :, 17, :] = self._I
			HHH[20, :, 18, :] = self._I

			HHH[38, :, 19, :] = np.cos(self._lam)*np.sin(self._theta)*self._Z
			HHH[38, :, 20, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[-5] += HHH.copy()
			self.mpo[-4] += HHH.copy()
			self.mpo[-3] += HHH.copy()
			self.mpo[-2] += HHH.copy()

			# -----

			HHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHH[21, :, 1, :] = self._X
			HHHH[22, :, 2, :] = self._Y
			HHHH[23, :, 21, :] = self._Z
			HHHH[24, :, 22, :] = self._Z

			HHHH[38, :, 23, :] = np.cos(self._lam)*np.sin(self._theta)*self._Z
			HHHH[38, :, 24, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[-4] += HHHH.copy()
			self.mpo[-3] += HHHH.copy()
			self.mpo[-2] += HHHH.copy()

			# -----

			HHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHH[25, :, 3, :] = self._X
			HHHHH[26, :, 3, :] = self._Y
			HHHHH[27, :, 25, :] = self._I
			HHHHH[28, :, 26, :] = self._I
			HHHHH[29, :, 27, :] = self._X
			HHHHH[30, :, 28, :] = self._Y
			HHHHH[31, :, 29, :] = self._I
			HHHHH[32, :, 30, :] = self._I

			HHHHH[38, :, 31, :] = np.cos(self._lam)*np.sin(self._theta)*self._Z
			HHHHH[38, :, 32, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[-6] += HHHHH.copy()
			self.mpo[-5] += HHHHH.copy()
			self.mpo[-4] += HHHHH.copy()
			self.mpo[-3] += HHHHH.copy()
			self.mpo[-2] += HHHHH.copy()

			# -----

			HHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHH[38, :, 11, :] = np.cos(self._lam)*np.sin(self._theta)*self._Z
			HHHHHH[38, :, 12, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			self.mpo[0] += HHHHHH.copy()

			# -----

			HHHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHHH[33, :, 3, :] = self._Z
			HHHHHHH[34, :, 33, :] = self._X
			HHHHHHH[35, :, 33, :] = self._Y

			HHHHHHH[38, :, 34, :] = np.cos(self._lam)*np.sin(self._theta)*self._X
			HHHHHHH[38, :, 35, :] = np.sin(self._lam)*np.cos(self._theta)*self._Y

			self.mpo[0] += HHHHHHH.copy()
			self.mpo[1] += HHHHHHH.copy()
			self.mpo[2] += HHHHHHH.copy()

			# -----

			HHHHHHHH = np.zeros([39, self.d, 39, self.d], dtype=np.complex128)

			HHHHHHHH[36, :, 7, :] = self._Z
			HHHHHHHH[37, :, 8, :] = self._Z

			HHHHHHHH[38, :, 36, :] = np.cos(self._lam)*np.sin(self._theta)*self._X
			HHHHHHHH[38, :, 37, :] = np.sin(self._lam)*np.cos(self._theta)*self._Y

			self.mpo[0] += HHHHHHHH.copy()
			self.mpo[1] += HHHHHHHH.copy()

		else:

			H = np.zeros([9, self.d, 9, self.d], dtype=np.complex128)

			H[0, :, 0, :] = self._I
			H[1, :, 0, :] = self._X
			H[2, :, 0, :] = self._Y
			H[3, :, 0, :] = self._Z

			H[4, :, 3, :] = self._X
			H[5, :, 3, :] = self._Y
			H[6, :, 4, :] = self._X
			H[7, :, 5, :] = self._Y

			H[8, :, 1, :] = np.cos(self._lam)*np.cos(self._theta)*self._X
			H[8, :, 2, :] = np.sin(self._lam)*np.sin(self._theta)*self._Y
			H[8, :, 6, :] = np.cos(self._lam)*np.sin(self._theta)*self._Z
			H[8, :, 7, :] = np.sin(self._lam)*np.cos(self._theta)*self._Z

			H[8, :, 8, :] = self._I

			self.mpo = [H.copy() for i in range(self.N)]


class iMPO(MPO):
	
	def ___init__(self, *args, **kwargs):
		super().___init__(*args, **kwargs)

		assert self.model["PBCs"] == False

	@override
	def _initLocal_A(self):

		H = np.cos(self._lam)*np.cos(self._theta)*np.kron(self._X, np.kron(self._X, np.kron(self._I, self._I)))/3
		H += np.cos(self._lam)*np.cos(self._theta)*np.kron(self._I, np.kron(self._X, np.kron(self._X, self._I)))/3
		H += np.cos(self._lam)*np.cos(self._theta)*np.kron(self._I, np.kron(self._I, np.kron(self._X, self._X)))/3
		H += np.cos(self._lam)*np.sin(self._theta)*np.kron(self._Y, np.kron(self._Y, np.kron(self._I, self._I)))/3
		H += np.cos(self._lam)*np.sin(self._theta)*np.kron(self._I, np.kron(self._Y, np.kron(self._Y, self._I)))/3
		H += np.cos(self._lam)*np.sin(self._theta)*np.kron(self._I, np.kron(self._I, np.kron(self._Y, self._Y)))/3
		H += np.sin(self._lam)*np.cos(self._theta)*np.kron(self._Z, np.kron(self._X, np.kron(self._X, self._Z)))
		H += np.sin(self._lam)*np.sin(self._theta)*np.kron(self._Z, np.kron(self._Y, np.kron(self._Y, self._Z)))

		H = H.reshape(self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d)
		
		self.local = [H.copy() for i in range(self.N)]

	@override
	def _initLocal_B(self):

		H = np.cos(self._lam)*np.cos(self._theta)*np.kron(self._X, np.kron(self._X, np.kron(self._I, self._I)))/3
		H += np.cos(self._lam)*np.cos(self._theta)*np.kron(self._I, np.kron(self._X, np.kron(self._X, self._I)))/3
		H += np.cos(self._lam)*np.cos(self._theta)*np.kron(self._I, np.kron(self._I, np.kron(self._X, self._X)))/3
		H += np.cos(self._lam)*np.sin(self._theta)*np.kron(self._Y, np.kron(self._Y, np.kron(self._I, self._I)))/3
		H += np.cos(self._lam)*np.sin(self._theta)*np.kron(self._I, np.kron(self._Y, np.kron(self._Y, self._I)))/3
		H += np.cos(self._lam)*np.sin(self._theta)*np.kron(self._I, np.kron(self._I, np.kron(self._Y, self._Y)))/3
		H += np.sin(self._lam)*np.sin(self._theta)*np.kron(self._Z, np.kron(self._X, np.kron(self._X, self._Z)))
		H += np.sin(self._lam)*np.cos(self._theta)*np.kron(self._Z, np.kron(self._Y, np.kron(self._Y, self._Z)))

		H = H.reshape(self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d)
		
		self.local = [H.copy() for i in range(self.N)]

	@override
	def _initLocal_C(self):

		H = np.cos(self._lam)*np.cos(self._theta)*np.kron(self._X, np.kron(self._X, np.kron(self._I, self._I)))/3
		H += np.cos(self._lam)*np.cos(self._theta)*np.kron(self._I, np.kron(self._X, np.kron(self._X, self._I)))/3
		H += np.cos(self._lam)*np.cos(self._theta)*np.kron(self._I, np.kron(self._I, np.kron(self._X, self._X)))/3
		H += np.sin(self._lam)*np.sin(self._theta)*np.kron(self._Y, np.kron(self._Y, np.kron(self._I, self._I)))/3
		H += np.sin(self._lam)*np.sin(self._theta)*np.kron(self._I, np.kron(self._Y, np.kron(self._Y, self._I)))/3
		H += np.sin(self._lam)*np.sin(self._theta)*np.kron(self._I, np.kron(self._I, np.kron(self._Y, self._Y)))/3
		H += np.cos(self._lam)*np.sin(self._theta)*np.kron(self._Z, np.kron(self._X, np.kron(self._X, self._Z)))
		H += np.sin(self._lam)*np.cos(self._theta)*np.kron(self._Z, np.kron(self._Y, np.kron(self._Y, self._Z)))

		H = H.reshape(self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d)
		
		self.local = [H.copy() for i in range(self.N)]

class Binder:

	def __init__(self, N, order):

		self.N = N
		self.order = order

		self._I = np.eye(self.order["operator"].shape[0])

		self._initMPO()

	def _initMPO(self):

		H = np.zeros([2, 2, 2, 2], dtype=np.complex128)

		H[0, :, 0, :] = self._I
		H[1, :, 0, :] = self._O

		H[1, :, 1, :] = self._I

		self.mpo = [H.copy() for i in range(self.N)]
		
		if self.order["type"] == "staggered":

			H_ = np.zeros([2, 2, 2, 2], dtype=np.complex128)

			H_[0, :, 0, :] = self._I
			H_[1, :, 0, :] = self._O

			H_[1, :, 1, :] = self._I

			for i in range(self.N):
				self.mpo[i] = H_.copy() if i % 2 == 0 else H.copy()



	