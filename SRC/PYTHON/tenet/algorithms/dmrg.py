import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln
import pickle
import time

from tenet.networks import EffectiveHamiltonian
from tenet.linalg import lanczos
from tenet.utils.decorators import *

__all__ = [
	'DMRGEngine',
	'iDMRGEngine',
]

class DMRGEngine:

	def __init__(self, mps, mpo, chi, sweeps, thresholds, convergence, precision, options, save):

		self._method = "DMRG"

		self.psi = mps
		self.H = mpo
		self.chi = chi

		self.num_sweeps_min = sweeps.get('min', 1)
		self.num_sweeps_max = sweeps.get('max', 10)

		self.tol_energy = thresholds.get('energy', 1e-6)
		self.tol_entropy = thresholds.get('entropy', 1e-6)
		self.tol_variance = thresholds.get('variance', 1e-6)

		self.converge_energy = convergence.get('energy', True)
		self.converge_entropy = convergence.get('entropy', True)
		self.converge_variance = convergence.get('variance', False)

		self.prec_svd = precision.get('svd', 1e-10)
		self.prec_eigen = precision.get('eigen', 1e-10)

		self.k = options.get('k', 1)
		self.krylov_dim = options.get('krylov_dim', 5)
		self.max_lanczos_iter = options.get('max_iter', 2)
		self.verbose = options.get('verbose', True)
		self.noise = options.get('noise', 0)
		self.num_sweeps_with_noise = options.get('noisy_sweeps', 1)

		self.save_every = save.get('every', 0)
		self.save_file = save.get('file', '')

		self._init_params()
		self._init_environment()

	def _list_private_attributes(self):
		return [attr for attr in dir(self) if attr.startswith('_') and not attr.startswith('__') and not callable(getattr(self, attr))]

	def _list_public_attributes(self):
		return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]

	def _init_params(self):
		self.N = self.psi.N
		self.d = self.psi.d

		self.E = [[] for k in range(self.k)]
		self.S = []

		self.dE = []
		self.dS = []
		self.dV = []

		self.times = []

		self.converged = False

		self.num_sweeps = 0

	def _init_environment(self):
		self.L = [None] * self.N
		self.R = [None] * self.N

		chi_L, chi_R = self.psi.B[0].shape[0], self.psi.B[-1].shape[2]
		mpo_dim = self.H.mpo[0].shape[0]
		self.L[0] = np.zeros([chi_L, mpo_dim, chi_L], dtype=np.complex128)
		self.R[-1] = np.zeros([chi_R, mpo_dim, chi_R], dtype=np.complex128)
		self.L[0][:, -1, :] = np.eye(chi_L)
		self.R[-1][:, 0, :] = np.eye(chi_R)

		for i in reversed(range(self.N - 1)):
			leg_links = [[-2, 1, 2, 3], [4, 2, 5], [-1, 1, 4], [-3, 3, 5]]
			self.R[i] = ncon([self.H.mpo[self._get_next_site(i, 1)], self.R[self._get_next_site(i, 1)], self.psi.B[self._get_next_site(i, 1)], np.conj(self.psi.B[self._get_next_site(i, 1)])], leg_links)

	def run(self):

		self._print_sys_params()

		if not self.converged:

			saved_last = 1

			self._use_own_lanczos = True
			self._stop = False
			self._update = True

			while (not self._stop or (self.num_sweeps < self.num_sweeps_min)) and (self.num_sweeps < self.num_sweeps_max):

				time, _ = self._do_one_sweep()
				self.times.append(time)
				self._finish_sweep()

				if saved_last == self.save_every:
					self._save_to_file(self.save_file)
					saved_last = 1
				else:
					saved_last += 1

				if self.num_sweeps == self.num_sweeps_with_noise:
					self.noise = 0

			if (self.k != 1) and self._stop:
				print("\t · CONVERGED ! Doing last sweep for excited energies.\n")
				self._use_own_lanczos = False
				time, _ = self._do_one_sweep()
				self.times.append(time)
				self._finish_sweep()

			self._update = False
			self._do_one_sweep()

			self.converged = True if self._stop else False

			if self.converged:
				self._lighten_space()

			print(f"------------------------------------------------------------ DONE")

		else:

			print(f"------------------------------------------------------------ ALREADY DONE")

	def update(self, updates: dict):

		for key, value in updates.items():
			if hasattr(self, key):
				setattr(self, key, value)

	def _lighten_space(self):

		self.psi.A = None

		self.L = None
		self.R = None

	def _save_to_file(self, file):

		with open(file, 'wb') as f:
			pickle.dump(self, f)

	def _print_sys_params(self):

		print(f"\n\t\t start {self._method} \n")
		print(f"\t | N : {self.N}")
		print(f"\t | chi : {self.chi}")
		print(f"\t | params :", self.H.params, "\n")

	def _finish_sweep(self):

		dV, dE, dS = self._check_convergence()
		self.cut_off = self._get_cut_off()

		print(f"Finished sweep {self.num_sweeps} in {self.times[-1]:.3f} s :")
		print(f"\t| E : {self._get_energy():.10f} \t dE : {dE:.3e}")
		print(f"\t| S : {self._get_entropy():.10f} \t dS : {dS:.3e}")
		print(f"\t| chi : {max(self.psi.get_chi())} \t err : {self.cut_off:.3e}")
		if self.converge_variance:
			print(f"\t| dV : {dV:.3e}")
		else:
			print("")

	def _get_energy(self):
		return self.E[0][self._middle_bond_in_sweep('left')]

	def _get_entropy(self):
		return self.S[self._middle_bond_in_sweep('left')]

	def _get_cut_off(self):
		return self.psi.get_cut_off(self._middle_bond_in_sweep('left'))

	def _print_update(self, i):

		if self._update:
			print(f"····· bond {i}·{self._get_next_site(i, 1)} | E : {self.E[0][-1]:.10f}")

	def _check_convergence(self):

		dV = 0
		if self.converge_variance:
			dV = self._compute_variance()
			self.dV.append(dV)

		dE = self._compute_energy_difference()
		self.dE.append(dE)

		dS = self._compute_entropy_difference()
		self.dS.append(dS)

		condition_variance = (dV < self.tol_variance) if self.converge_variance else True
		condition_energy = (dE < self.tol_energy) if self.converge_energy else True
		condition_entropy = (dS < self.tol_entropy) if self.converge_entropy else True

		if condition_variance and condition_energy and condition_entropy:
			self._stop = True

		return dV, dE, dS

	def _compute_energy_difference(self):
		return np.abs(self.E[0][self._middle_bond_in_sweep('left')] - self.E[0][self._middle_bond_in_sweep('right')])

	def _compute_entropy_difference(self):
		return np.abs(self.S[self._middle_bond_in_sweep('left')] - self.S[self._middle_bond_in_sweep('right')])

	def _middle_bond_in_sweep(self, direction):
		if direction == 'right':
			return - self.N // 2 - (self.N - 1)
		elif direction == 'left':
			return - self.N // 2

	def _svd(self, tensor):
		# returns u, s, v as tuple
		return np.linalg.svd(tensor, full_matrices=False)

	def _measure_bond(self, i, direction, boundary):
		if not boundary and self._update:
			self._measure_entropy(i, direction)
			self._measure_energy()

	def _measure_entropy(self, i, direction):
		if direction == 'right':
			self.S.append(self.psi.compute_entropy(self._get_next_site(i, 1)))
		elif direction == 'left':
			self.S.append(self.psi.compute_entropy(i))

	def _measure_energy(self):
		for k in range(len(self._E)):
			self.E[k].append(self._E[k])

	@timed
	def _do_one_sweep(self):
		direction = 'right'
		for i in range(self.N):
			i = self._get_site(i)
			self._update_bond(i, direction, self._is_boundary(i, direction))
			self._measure_bond(i, direction, self._is_boundary(i, direction))
			if self.verbose:
				self._print_update(i)

		direction = 'left'
		for i in reversed(range(self.N)):
			i = self._get_site(i)
			self._update_bond(i, direction, self._is_boundary(i, direction))
			self._measure_bond(i, direction, self._is_boundary(i, direction))
			if self.verbose:
				self._print_update(self._get_next_site(i, -1))

		self.num_sweeps += 1 if self._update else 0

	def _is_boundary(self, i, direction):
		return ((i == (self.N - 1)) and (direction == 'right')) or ((i == 0) and (direction == 'left'))

	def _get_site(self, i):
		return i

	def _get_next_site(self, i, step):
		return (i + step)

	def _set_update_chi(self, i, direction, boundary):
		if boundary:
			if direction == 'right':
				self._chi_L = self.psi.B[i].shape[0]
				self._chi_R = self.psi.B[i].shape[2]
			elif direction == 'left':
				self._chi_L = self.psi.A[i].shape[0]
				self._chi_R = self.psi.A[i].shape[2]
		else:
			if direction == 'right':
				self._chi_L = self.psi.B[i].shape[0]
				self._chi_R = self.psi.B[self._get_next_site(i, 1)].shape[2]
			elif direction == 'left':
				self._chi_L = self.psi.A[self._get_next_site(i, -1)].shape[0]
				self._chi_R = self.psi.A[i].shape[2]

	def _get_local_tensor(self, i, direction, boundary):
		if boundary:
			# returns a rank 2 tensor
			if direction == 'right':
				leg_links = [[-1, 1], [1, -2, -3]]
				tensor = ncon([self.psi.weight[i], self.psi.B[i]], leg_links).reshape(self._chi_L * self.d, self._chi_R)
			elif direction == 'left':
				leg_links = [[-1, -2, 1], [1, -3]]
				tensor = ncon([self.psi.A[i], self.psi.weight[self._get_next_site(i, 1)]], leg_links).reshape(self._chi_L, self.d * self._chi_R)
		else:
			# returns a rank 1 tensor
			if direction == 'right':
				leg_links = [[-1, 1], [1, -2, 2], [2, -3, -4]]
				tensor = ncon([self.psi.weight[i], self.psi.B[i], self.psi.B[self._get_next_site(i, 1)]], leg_links).reshape(self._chi_L * self.d * self.d * self._chi_R)
			elif direction == 'left':
				leg_links = [[-1, -2, 1], [1, -3, 2], [2, -4]]
				tensor = ncon([self.psi.A[self._get_next_site(i, -1)], self.psi.A[i], self.psi.weight[self._get_next_site(i, 1)]], leg_links).reshape(self._chi_L * self.d * self.d * self._chi_R)
		return tensor

	def _set_updated_tensor_infinite(self, i):
		pass
	def _set_environment_infinite(self, i):
		pass

	def _set_updated_tensor(self, i, direction, boundary, svd_tuple):
		u, s, v = svd_tuple
		if boundary:
			if direction == 'right':
				self.psi.A[i] = u.reshape(self._chi_L, self.d, self._chi_R)
				self.psi.weight[self._get_next_site(i, 1)] = np.diag(s) @ v
			elif direction == 'left':
				self.psi.B[i] = v.reshape(self._chi_L, self.d, self._chi_R)
				self.psi.weight[i] = u @ np.diag(s)
		else:
			if direction == 'right':
				self.psi.A[i] = u[:, :self._chi_trunc].reshape(self._chi_L, self.d, self._chi_trunc)
				self.psi.B[self._get_next_site(i, 1)] = v[:self._chi_trunc, :].reshape(self._chi_trunc, self.d, self._chi_R)
				self.psi.weight[self._get_next_site(i, 1)] = np.diag(s[:self._chi_trunc]) / np.linalg.norm(s[:self._chi_trunc])
				self._set_updated_tensor_infinite(i)
			elif direction == 'left':
				self.psi.A[self._get_next_site(i, -1)] = u[:, :self._chi_trunc].reshape(self._chi_L, self.d, self._chi_trunc)
				self.psi.B[i] = v[:self._chi_trunc, :].reshape(self._chi_trunc, self.d, self._chi_R)
				self.psi.weight[i] = np.diag(s[:self._chi_trunc]) / np.linalg.norm(s[:self._chi_trunc])

	def _set_cut_off(self, s):
		self._chi_trunc = min(self.chi, np.sum(s > self.prec_svd))

	def _set_environment(self, i, direction):
		if direction == 'right':
			legLinks = [[1, 2, 4], [2, 3, -2, 5], [1, 3, -1], [4, 5, -3]]
			self.L[self._get_next_site(i, 1)] = ncon([self.L[i], self.H.mpo[i], self.psi.A[i], np.conjugate(self.psi.A[i])], legLinks)
			self._set_environment_infinite(i)
		elif direction == 'left':
			leg_links = [[-2, 3, 2, 5], [1, 2, 4], [-1, 3, 1], [-3, 5, 4]]
			self.R[self._get_next_site(i, -1)] = ncon([self.H.mpo[i], self.R[i], self.psi.B[i], np.conjugate(self.psi.B[i])], leg_links)

	def _update_bond(self, i, direction, boundary):
		self._set_update_chi(i, direction, boundary)
		tensor = self._get_local_tensor(i, direction, boundary)
		if not boundary:
			tensor =  self._perform_update(tensor, i, direction)
		assert len(tensor.shape) == 2
		svd_tuple = self._svd(tensor)
		if not boundary:
			self._set_cut_off(svd_tuple[1])
		self._set_updated_tensor(i, direction, boundary, svd_tuple)
		if not boundary:
			self._set_environment(i, direction)

	def _perform_update(self, tensor, i, direction):
		# returns a rank 2 tensor
		if self._update:

			if self._use_own_lanczos:
				k = 1
				if direction == 'right':
					lin = EffectiveHamiltonian(self.L[i].copy(), self.H.mpo[i].copy(), self.H.mpo[self._get_next_site(i, 1)].copy(), self.R[self._get_next_site(i, 1)].copy(), self.noise)
				elif direction == 'left':
					lin = EffectiveHamiltonian(self.L[self._get_next_site(i, -1)].copy(), self.H.mpo[self._get_next_site(i, -1)].copy(), self.H.mpo[i].copy(), self.R[i].copy(), self.noise)
				e, v = lanczos.restarted_lanczos(H=lin, v=tensor, m=self.krylov_dim, k=k, max_restarts=self.max_lanczos_iter, tol=self.prec_eigen)
			else:
				k = self.k
				if direction == 'right':
					lin = EffectiveHamiltonian(self.L[i], self.H.mpo[i], self.H.mpo[self._get_next_site(i, 1)], self.R[self._get_next_site(i, 1)], self.noise)
				elif direction == 'left':
					lin = EffectiveHamiltonian(self.L[self._get_next_site(i, -1)], self.H.mpo[self._get_next_site(i, -1)], self.H.mpo[i], self.R[i], self.noise)
				e, v = ln.eigsh(lin, v0=tensor, k=k, which='SA', return_eigenvectors=True, tol=self.prec_eigen)

			tensor = v[:, 0]
			self._E = e

		return tensor.reshape(self._chi_L * self.d, self.d * self._chi_R)

	def _compute_variance(self):

		Hl = self.L[0]
		Hr = self.R[-1]

		leg_links = [[1, 3, -1], [1, 2, 4], [2, 3, -2, 5], [4, 6, 7], [6, 5, -3, 8], [7, 8, -4]]
		h_squared = ncon([self.psi.B[0], Hl, self.H.mpo[0], Hl, self.H.mpo[0], np.conjugate(self.psi.B[0])], leg_links)
		leg_links = [[1, 3, -1], [1, 2, 4], [2, 3, -2, 5], [4, 5, -3]]
		squared_h = ncon([self.psi.B[0], Hl, self.H.mpo[0], np.conjugate(self.psi.B[0])], leg_links)

		for i in range(1, self.N-1):
			leg_links = [[1, 2, 4, 6], [1, 3, -1], [2, 3, -2, 5], [4, 5, -3, 7], [6, 7, -4]]
			h_squared = ncon([h_squared, self.psi.B[i], self.H.mpo[i], self.H.mpo[i], np.conjugate(self.psi.B[i])], leg_links)
			leg_links = [[1, 3, -1], [1, 2, 4], [2, 3, -2, 5], [4, 5, -3]]
			squared_h = ncon([self.psi.B[i], squared_h, self.H.mpo[i], np.conjugate(self.psi.B[i])], leg_links)

		leg_links = [[9, 10, 11, 12], [9, 3, 1], [10, 3, 2, 5], [1, 2, 4], [11, 5, 6, 8], [4, 6, 7], [12, 8, 7]]
		h_squared = ncon([h_squared, self.psi.B[-1], self.H.mpo[-1], Hr, self.H.mpo[-1], Hr, np.conjugate(self.psi.B[-1])], leg_links)
		leg_links = [[6, 3, 1], [6, 7, 8], [7, 3, 2, 5], [1, 2, 4], [8, 5, 4]]
		squared_h = ncon([self.psi.B[-1], squared_h, self.H.mpo[-1], Hr, np.conjugate(self.psi.B[-1])], leg_links)

		var = h_squared - squared_h**2

		return np.abs(var)

class iDMRGEngine(DMRGEngine):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		assert self.k == 1

		self._method = "iDMRG"

		self._inf_E = [0]
		self._inf_S = [0]

	@timed
	@override
	def _do_one_sweep(self):

		direction = 'right'
		sites = [i for i in range(self.N)]
		for i in sites:
			i = self._get_site(i)
			self._update_bond(i, direction, self._is_boundary(i, direction))
			self._measure_bond(i, direction, self._is_boundary(i, direction))
			if self.verbose:
				self._print_update(i)

		direction = 'right'
		sites = [(i - 1) for i in reversed(range(self.N))]
		for i in sites:
			i = self._get_site(i)
			self._update_bond(i, direction, self._is_boundary(i, direction))
			self._measure_bond(i, direction, self._is_boundary(i, direction))
			if self.verbose:
				self._print_update(i)

		self._inf_E.append(self._get_energy())
		self._inf_S.append(self._get_entropy())

		self.num_sweeps += 1 if self._update else 0

	@override
	def _is_boundary(self, i, direction):
		return False

	@override
	def _get_site(self, i):
		return i % self.N

	@override
	def _get_next_site(self, i, step):
		return (i + step) % self.N

	@override
	def _set_updated_tensor_infinite(self, i):
		legLinks = [[-1, 1], [1, -2, 2], [2, -3]]
		self.psi.B[i] = ncon([np.diag(np.diag(self.psi.weight[i])**(-1)), self.psi.A[i], self.psi.weight[self._get_next_site(i, 1)]], legLinks)

	@override
	def _set_environment_infinite(self, i):
		legLinks = [[1, 2, 4], [-2, 3, 2, 5], [-1, 3, 1], [-3, 5, 4]]
		self.R[i] = ncon([self.R[self._get_next_site(i, 1)], self.H.mpo[self._get_next_site(i, 1)], self.psi.B[self._get_next_site(i, 1)], np.conjugate(self.psi.B[self._get_next_site(i, 1)])], legLinks)

	@override
	def _middle_bond_in_sweep(self, direction):
		if direction == 'right':
			return - (self.N // 2 + 1) - self.N
		elif direction == 'left':
			return - (self.N // 2 + 1)

	@override
	def _get_energy(self):
		return np.mean(self.psi.compute_bond_expectation_value_MPS(self.H.local))

	@override
	def _get_entropy(self):
		return np.max(self.psi.compute_entropy())

	@override
	def _compute_energy_difference(self):
		return np.abs(self._inf_E[-1] - self._inf_E[-2])

	@override
	def _compute_entropy_difference(self):
		return np.abs(self._inf_S[-1] - self._inf_S[-2])