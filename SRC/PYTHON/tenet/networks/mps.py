import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln

from tenet.linalg import SUIrrep
from tenet.utils.decorators import *
from tenet.networks.mpo import Binder

__all__ = [
	'MPS',
	'iMPS',
]

class MPS:

	def __init__(self, N, p, state):
		"""
		Initialize the MPS object with the given parameters.

		Args:
		N (int): Number of sites in the system.
		p (object): Parameters object containing dimension and generators.
		state (str): State of the system, can be "up", "down", or "neel".
		"""
		self.N = N
		self.p = p
		self.state = state

		self.d = p.dim
		self.T = p.generators

		if p.N == 2:
			self._X, self._Y, self._Z = [self.T[i] for i in range(3)]

		self._init_MPS()

	def _init_MPS(self):
		"""
		Initialize the Matrix Product State (MPS) based on the specified state.
		"""
		if self.state == "up":
			self.B = [np.array([1] + [0 for i in range(self.d - 1)], dtype=np.complex128).reshape(1, self.d, 1) for i in range(self.N)]
			self.weight = [np.eye(1) for i in range(self._get_num_bonds())]

		elif self.state == "down":
			self.B = [np.array([0 for i in range(self.d - 1)] + [1], dtype=np.complex128).reshape(1, self.d, 1) for i in range(self.N)]
			self.weight = [np.eye(1) for i in range(self._get_num_bonds())]

		elif self.state == "neel":
			self.B = [None for i in range(self.N)]
			for i in range(0, self.N, 2):
				self.B[i] = np.array([1] + [0 for i in range(self.d - 1)], dtype=np.complex128).reshape(1, self.d, 1)
			for i in range(1, self.N, 2):
				self.B[i] = np.array([0 for i in range(self.d - 1)] + [1], dtype=np.complex128).reshape(1, self.d, 1)
			self.weight = [np.eye(1) for i in range(self._get_num_bonds())]

		self.A = self.B.copy()

	def _get_num_bonds(self):
		"""
		Calculate the number of bonds in the system.

		Returns:
		int: Number of bonds in the system.
		"""
		return (self.N + 1)

	
	def get_chi(self):
		"""
		Calculate the number of elements in each weight matrix.

		Returns:
		list: A list containing the number of elements in each weight matrix.
		"""

		return [self.weight[i].shape[0] for i in range(self._get_num_bonds())]

	def get_cut_off(self, i):
		"""
		Get the minimum value along the diagonal of a specific weight matrix.

		Args:
		i (int): Index of the weight matrix.

		Returns:
		float: The minimum value along the diagonal of the specified weight matrix.
		"""

		return min(np.diag(self.weight[i]))

	def compute_entropy(self, i = None):
		"""
		Compute the entropy of a weight matrix or all weight matrices.

		Args:
		i (int, optional): Index of the weight matrix. If None, compute entropy for all weight matrices.

		Returns:
		float or list: The entropy value of the specified weight matrix or a list of entropy values for all weight matrices.
		"""

		if i is not None:
			s = np.diag(self.weight[i])
			return -np.sum(s ** 2 * np.log(s ** 2)).real
		else:
			S = []
			for i in range(self._get_num_bonds()):
				s = np.diag(self.weight[i])
				S.append(-np.sum(s ** 2 * np.log(s ** 2)).real)
			return np.array(S)

	def compute_bond_expectation_value_MPS(self, local_op):
		"""
		Computes the bond expectation values for all sites in the MPS.

		Args:
		self: The MPS object.
		local_op: The local operator to compute the expectation value with.

		Returns:
		A list containing the bond expectation values for all sites.
		"""
		result = []

		for i in range(self._get_num_bonds()):
			result.append(self.compute_bond_expectation_value(local_op, i))

		return result

	def compute_bond_expectation_value(self, local_op, i):
		"""
		Computes the bond expectation value for a specific site in the MPS.

		Args:
		self: The MPS object.
		local_op: The local operator to compute the expectation value with.
		i: The index of the site to compute the expectation value for.

		Returns:
		The bond expectation value for the specified site.
		"""
		num_legs = len(local_op[i].shape)

		if num_legs == 2:
			legLinks = [[-1, 1], [1, -2, -3]]
			psi = ncon([self.weight[i], self.B[i]], legLinks)

			legLinks = [[1, 2, 4], [2, 3], [1, 3, 4]]
			result = ncon([psi, local_op[i], np.conjugate(psi)], legLinks)

		if num_legs == 4:
			iNext = (i + 1) % self.N

			legLinks = [[-1, 1], [1, -2, 2], [2, -3, -4]]
			psi = ncon([self.weight[i], self.B[i], self.B[iNext]], legLinks)

			legLinks = [[1, 2, 4, 6], [2, 4, 3, 5], [1, 3, 5, 6]]
			result = ncon([psi, local_op[i], np.conjugate(psi)], legLinks)

		if num_legs == 6:
			iNext = (i + 1) % self.N
			iNextNext = (i + 2) % self.N

			legLinks = [[-1, 1], [1, -2, 2], [2, -3, 3], [3, -4, -5]]
			psi = ncon([self.weight[i], self.B[i], self.B[iNext], self.B[iNextNext]], legLinks)

			legLinks = [[1, 2, 4, 6, 8], [2, 4, 6, 3, 5, 7], [1, 3, 5, 7, 8]]
			result = ncon([psi, local_op[i], np.conjugate(psi)], legLinks)

		elif num_legs == 8:
			iNext = (i + 1) % self.N
			iNextNext = (i + 2) % self.N
			iNextNextNext = (i + 3) % self.N

			legLinks = [[-1, 1], [1, -2, 2], [2, -3, 3], [3, -4, 4], [4, -5, -6]]
			psi = ncon([self.weight[i], self.B[i], self.B[iNext], self.B[iNextNext], self.B[iNextNextNext]], legLinks)

			legLinks = [[1, 2, 4, 6, 8, 10], [2, 4, 6, 8, 3, 5, 7, 9], [1, 3, 5, 7, 9, 10]]
			result = ncon([psi, local_op[i], np.conjugate(psi)], legLinks)

		return np.real_if_close(result)

	def compute_overlap_R(self, local_op, psiR):
		"""
		Compute the overlap between the current state and the right wavefunction.

		Args:
		self: Current object instance.
		local_op: Local operator to apply.
		psiR: Right wavefunction.

		Returns:
		overlapR: Array containing the overlap values.
		"""

		overlapR = np.zeros((self.N, psiR.shape[1]), dtype=np.complex128)

		for site in range(self.N):

			legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
			state = ncon([self.weight[site], self.B[site], local_op, self.weight[site], np.conjugate(self.B[site])], legLinks)

			for i in range(site+1, self.N):

				legLinks = [[1, 3], [1, 2, -1], [3, 2, -2]]
				state = ncon([state, self.B[i], np.conjugate(self.B[i])], legLinks)

			for i in range(psiR.shape[1]):

				legLinks = [[1, 2], [1, 2]]
				overlapR[site, i] = ncon([state, psiR[:, i].reshape(self.B[-1].shape[2], self.B[-1].shape[2])], legLinks)

		return overlapR

	def compute_overlap_L(self, local_op, psiL):
		"""
		Compute the overlap between the current state and the left wavefunction.

		Args:
		self: Current object instance.
		local_op: Local operator to apply.
		psiL: Left wavefunction.

		Returns:
		overlapL: Array containing the overlap values.
		"""

		overlapL = np.zeros((self.N, psiL.shape[1]), dtype=np.complex128)

		for site in range(self.N):

			legLinks = [[-1, 2, 1], [2, 3], [-2, 3, 1]]
			state = ncon([self.B[site], local_op, np.conjugate(self.B[site])], legLinks)

			for i in range(site-1, -1, -1):

				legLinks = [[-1, 2, 1], [-2, 2, 3], [1, 3]]
				state = ncon([self.B[i], np.conjugate(self.B[i]), state], legLinks)

			for i in range(psiL.shape[1]):

				legLinks = [[1, 2], [1, 2]]
				overlapL[site, i] = ncon([psiL[:, i].reshape(self.B[0].shape[0], self.B[0].shape[0]), state], legLinks)

		return overlapL

	def compute_ev_2(self, local_op, start):

		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[start], self.B[start], local_op[0], self.weight[start], np.conjugate(self.B[start])], legLinks)
		
		legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
		res = ncon([state, self.B[start+1], local_op[1], np.conjugate(self.B[start+1])], legLinks)
		
		return np.real_if_close(res)

	def compute_ev_3(self, local_op, start):

		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[start], self.B[start], local_op[0], self.weight[start], np.conjugate(self.B[start])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+1], local_op[1], np.conjugate(self.B[start+1])], legLinks)
		
		legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
		res = ncon([state, self.B[start+2], local_op[2], np.conjugate(self.B[start+2])], legLinks)
		
		return np.real_if_close(res)

	def compute_ev_4(self, local_op, start):

		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[start], self.B[start], local_op[0], self.weight[start], np.conjugate(self.B[start])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+1], local_op[1], np.conjugate(self.B[start+1])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+2], local_op[2], np.conjugate(self.B[start+2])], legLinks)
		
		legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
		res = ncon([state, self.B[start+3], local_op[3], np.conjugate(self.B[start+3])], legLinks)
		
		return np.real_if_close(res)

	def compute_ev_6(self, local_op, start):

		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[start], self.B[start], local_op[0], self.weight[start], np.conjugate(self.B[start])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+1], local_op[1], np.conjugate(self.B[start+1])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+2], local_op[2], np.conjugate(self.B[start+2])], legLinks)
				
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+3], local_op[3], np.conjugate(self.B[start+3])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+4], local_op[4], np.conjugate(self.B[start+4])], legLinks)
		
		legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
		res = ncon([state, self.B[start+5], local_op[5], np.conjugate(self.B[start+5])], legLinks)
		
		return np.real_if_close(res)

	def compute_ev_8(self, local_op, start):

		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[start], self.B[start], local_op[0], self.weight[start], np.conjugate(self.B[start])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+1], local_op[1], np.conjugate(self.B[start+1])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+2], local_op[2], np.conjugate(self.B[start+2])], legLinks)
				
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+3], local_op[3], np.conjugate(self.B[start+3])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+4], local_op[4], np.conjugate(self.B[start+4])], legLinks)
						
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+5], local_op[5], np.conjugate(self.B[start+5])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+6], local_op[6], np.conjugate(self.B[start+6])], legLinks)
		
		legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
		res = ncon([state, self.B[start+7], local_op[7], np.conjugate(self.B[start+7])], legLinks)
		
		return np.real_if_close(res)

	def compute_ev_10(self, local_op, start):

		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[start], self.B[start], local_op[0], self.weight[start], np.conjugate(self.B[start])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+1], local_op[1], np.conjugate(self.B[start+1])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+2], local_op[2], np.conjugate(self.B[start+2])], legLinks)
				
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+3], local_op[3], np.conjugate(self.B[start+3])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+4], local_op[4], np.conjugate(self.B[start+4])], legLinks)
						
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+5], local_op[5], np.conjugate(self.B[start+5])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+6], local_op[6], np.conjugate(self.B[start+6])], legLinks)
								
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+7], local_op[7], np.conjugate(self.B[start+7])], legLinks)
		
		legLinks = [[1, 4], [1, 2, -1], [2, 3], [4, 3, -2]]
		state = ncon([state, self.B[start+8], local_op[8], np.conjugate(self.B[start+8])], legLinks)
		
		legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
		res = ncon([state, self.B[start+9], local_op[9], np.conjugate(self.B[start+9])], legLinks)
		
		return np.real_if_close(res)

	def compute_spin(self):
		"""
		Compute the spin values for each site.

		Args:
		self: Current object instance.

		Returns:
		sX: Array containing the X component of spin values.
		sY: Array containing the Y component of spin values.
		sZ: Array containing the Z component of spin values.
		"""

		sX = []
		sY = []
		sZ = []

		for i in range(self.N):

			legLinks = [[1, 2], [2, 3, 6], [3, 4], [1, 5], [5, 4, 6]]
			sX.append(ncon([self.weight[i], self.B[i], self._X, self.weight[i], np.conjugate(self.B[i])], legLinks))
			sY.append(ncon([self.weight[i], self.B[i], self._Y, self.weight[i], np.conjugate(self.B[i])], legLinks))
			sZ.append(ncon([self.weight[i], self.B[i], self._Z, self.weight[i], np.conjugate(self.B[i])], legLinks))

		return np.real_if_close(sX), np.real_if_close(sY), np.real_if_close(sZ)
		
	def compute_correlation(self, local_op, j, s):
		"""
		Compute the correlation of a quantum system.

		Args:
		self: The quantum system object.
		local_op: List of local operators.
		j: Starting point.
		s: Separation.

		Returns:
		List of correlation values.
		"""
		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[j], self.B[j], local_op[0], self.weight[j], np.conjugate(self.B[j])], legLinks)

		for i in range(j + 1, s):

			legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
			res.append(ncon([state, self.B[i], local_op[1], np.conjugate(self.B[i])], legLinks))

			legLinks = [[1, 3], [1, 2, -1], [3, 2, -2]]
			state = ncon([state, self.B[i], np.conjugate(self.B[i])], legLinks)

		return np.real_if_close(res)

class iMPS(MPS):

	def __init__(self, *args, **kwargs):
		"""
		Constructor for the class.
		
		Parameters:
		*args: Variable length argument list.
		**kwargs: Arbitrary keyword arguments.
		"""
		super().__init__(*args, **kwargs)

	@override
	def _get_num_bonds(self):
		"""
		Get the number of bonds.
		
		Returns:
		int: Number of bonds.
		"""
		return self.N

	@override
	def get_cut_off(self, i):
		"""
		Get the cut-off value for a specific index.
		
		Parameters:
		i (int): Index to retrieve the cut-off value for.
		
		Returns:
		float: Cut-off value.
		"""
		return max([min(np.diag(self.weight[i])) for i in range(self._get_num_bonds())]).item()

	@override
	def compute_correlation(self, local_op, r):
		"""
		Compute the correlation of a quantum system.

		Args:
		self: The quantum system object.
		local_op: List of local operators.
		r: Number of iterations.

		Returns:
		List of correlation values.
		"""
		res = []

		legLinks = [[1, 2], [2, 3, -1], [3, 4], [1, 5], [5, 4, -2]]
		state = ncon([self.weight[0], self.B[0], local_op[0], self.weight[0], np.conjugate(self.B[0])], legLinks)

		for i in range(1, self.N):

			legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
			res.append(ncon([state, self.B[i], local_op[1], np.conjugate(self.B[i])], legLinks))

			legLinks = [[1, 3], [1, 2, -1], [3, 2, -2]]
			state = ncon([state, self.B[i], np.conjugate(self.B[i])], legLinks)

		for j in range(r-1):

			for i in range(self.N):

				legLinks = [[1, 4], [1, 2, 5], [2, 3], [4, 3, 5]]
				res.append(ncon([state, self.B[i], local_op[1], np.conjugate(self.B[i])], legLinks))

				legLinks = [[1, 3], [1, 2, -1], [3, 2, -2]]
				state = ncon([state, self.B[i], np.conjugate(self.B[i])], legLinks)

		return np.real_if_close(res)