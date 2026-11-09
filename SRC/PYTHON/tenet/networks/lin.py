import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln

__all__ = [
	'TransferMatrix',
	'Hamiltonian',
	'EffectiveHamiltonian'
]

class EffectiveHamiltonian(ln.LinearOperator):

	def __init__(self, L, Hl, Hr, R, noise=0):
		
		self.L = L
		self.R = R
		self.Hl = Hl + noise * np.random.randn(*Hl.shape)
		self.Hr = Hr + noise * np.random.randn(*Hr.shape)
		
		self.chi_L = L.shape[0]
		self.chi_R = R.shape[0]
		self.d_L = Hl.shape[1]
		self.d_R = Hr.shape[1]

		self.dtype = Hr.dtype
		self.shape = (self.chi_L * self.d_L * self.d_R * self.chi_R, self.chi_L * self.d_L * self.d_R * self.chi_R)


	def _matvec(self, mps):

		mps = mps.reshape(self.chi_L, self.d_L, self.d_R, self.chi_R)
		
		legLinks = [[1, 3, 5, 7], [1, 2, -1], [2, 3, 4, -2], [4, 5, 6, -3], [7, 6, -4]]
		res = ncon([mps, self.L, self.Hl, self.Hr, self.R], legLinks).reshape(self.chi_L * self.d_L * self.d_R * self.chi_R)

		return res

class TransferMatrix(ln.LinearOperator):
	
	def __init__(self, mps, side="L"):
		
		self.mps = mps
		self.side = side

		self.dtype = mps.B[0].dtype
		self.shape = (mps.B[0].shape[0] * mps.B[0].shape[0], mps.B[-1].shape[2] * mps.B[-1].shape[2])

	def getT(self):
		
		legLinks = [[-1, 1, -3], [-2, 1, -4]]
		res = ncon([self.mps.B[0], np.conjugate(self.mps.B[0])], legLinks)
	
		for i in range(1, self.mps.N):

			legLinks = [[-1, -2, 1, 3], [1, 2, -3], [3, 2, -4]]
			res = ncon([res, self.mps.B[i], np.conjugate(self.mps.B[i])], legLinks)
		
		return res.reshape(res.shape[0]*res.shape[1], res.shape[2]*res.shape[3])

	def _matvec(self, vec):

		if self.side == "L":
		
			res = vec.reshape(self.mps.B[0].shape[0], self.mps.B[0].shape[0])
			for i in range(self.mps.N):

				legLinks = [[1, 2], [1, 3, -1], [2, 3, -2]]
				res = ncon([res, self.mps.B[i], np.conjugate(self.mps.B[i])], legLinks)
		
		elif self.side == "R":
		
			res = vec.reshape(self.mps.B[-1].shape[-1], self.mps.B[-1].shape[-1])
			for i in range(self.mps.N-1, -1, -1):

				legLinks = [[1, 2], [-1, 3, 1], [-2, 3, 2]]
				res = ncon([res, self.mps.B[i], np.conjugate(self.mps.B[i])], legLinks)

		return res

class Hamiltonian(ln.LinearOperator):

	def __init__(self, model, params):

		self.model = model
		self.params = params

		self.init_system()

	def init_system(self):

		X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
		Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
		Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
		I = np.eye(2)

		self.s = np.array([I, X, Y, Z])

		self.N = self.params["N"]
		self.d = 2

		self.PBCs = self.model['BCs'] == "PBC"

	def get_full_sigma(self, i, j):

		return np.kron(np.identity(self.d**(j % self.N)), np.kron(self.s[i], np.identity(self.d**((self.N - j - 1) % self.N))))

	def get_full_matrix(self):

		H = np.zeros((self.d**self.N, self.d**self.N), dtype=np.complex128)

		for i in range(self.N-3+3*self.PBCs):

			if self.model['name'] == "A":

				H += np.sin(self.params['lam']) * np.cos(self.params['theta']) * self.get_full_sigma(3, i) @ self.get_full_sigma(1, i+1) @ self.get_full_sigma(1, i+2) @ self.get_full_sigma(3, i+3)
				H += np.sin(self.params['lam']) * np.sin(self.params['theta']) * self.get_full_sigma(3, i) @ self.get_full_sigma(2, i+1) @ self.get_full_sigma(2, i+2) @ self.get_full_sigma(3, i+3)

			elif self.model['name'] == "B":
				
				H += np.sin(self.params['lam']) * np.sin(self.params['theta']) * self.get_full_sigma(3, i) @ self.get_full_sigma(1, i+1) @ self.get_full_sigma(1, i+2) @ self.get_full_sigma(3, i+3)
				H += np.sin(self.params['lam']) * np.cos(self.params['theta']) * self.get_full_sigma(3, i) @ self.get_full_sigma(2, i+1) @ self.get_full_sigma(2, i+2) @ self.get_full_sigma(3, i+3)

		for i in range(self.N-1+self.PBCs):

			H += np.cos(self.params['lam']) * np.cos(self.params['theta']) * self.get_full_sigma(1, i) @ self.get_full_sigma(1, i+1)
			H += np.cos(self.params['lam']) * np.sin(self.params['theta']) * self.get_full_sigma(2, i) @ self.get_full_sigma(2, i+1)

		return H

	def _matvec(self):
		pass
