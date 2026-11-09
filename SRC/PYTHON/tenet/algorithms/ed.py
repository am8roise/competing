import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as ln

from tenet.algorithms.core.ed import build_hamiltonian
from tenet.networks.lin import Hamiltonian

__all__ = [
	'diagonalize',
	'diagonalize_exact'
]

def _make_sector(blocks, values):
	"""
	Build a sector dict from active blocks and loop values.
	"""
	return {k: v for k, v in values.items() if blocks.get(k, False)}

def _valid_sector(sector, N):
	"""
	Filter out sectors that are incompatible with U_X.
	"""
	if 'X' in sector and 'M' in sector:
		if sector['M'] != N // 2:
			return False
	return True

def diagonalize(model, params, tweaks, blocks):
	"""
	Diagonalizes the Hamiltonian for a given model and parameters.

	Supported block labels:
		- K : momentum
		- M : exact fermion number / magnetisation
		- P : fermion parity (0 even, 1 odd)

	Important:
		If M is used, P is redundant because parity is already fixed by M.
		In that case, this function automatically ignores P and only uses M.

	Args:
		model (dict): Dictionary containing model information.
		params (dict): Dictionary containing parameters such as 'N', 'lam', and 'theta'.
		tweaks (dict): Dictionary containing options like 'n_eigen' and 'print_time'.
		blocks (dict): Dictionary containing the sectors to split the Hamiltonian into (M and/or K and/or P).

	Returns:
		dict: A dictionary with sectors as keys and sorted eigenvalues as values.
	"""
	N = params.get('N', 0)
	n_eigen = tweaks.get('n_eigen', 10)
	print_time = tweaks.get('print_time', False)

	energies = {}

	if blocks['M'] and blocks['P']:
		print("\n[warning] Both 'M' and 'P' requested, hence using 'M' only, because parity is fixed inside each M sector.")

		blocks['P'] = False

	Ms = range(N + 1) if blocks['M'] else range(1)
	Ks = range(N) if blocks['K'] else range(1)
	Ps = [0, 1] if blocks['P'] else range(1)

	for M in Ms:
		for K in Ks:
			for P in Ps:
			
				values = {'M': M, 'K': K, 'P': P}
				sector = _make_sector(blocks, values)

				if print_time:
					print(f"\n\t ---> {sector}")
					start_time = time.time()

				H, basis = build_hamiltonian(model, params, sector)

				if print_time:
					total_time = time.time() - start_time
					print(f"\t\t• built Hamiltonian in {total_time:.5f} s --- size {H.shape[0]}")

				if H.shape[0] == 0:
					continue

				if print_time:
					start_time = time.time()

				if H.shape[0] < n_eigen // 2:
					eigvals = ln.eigh(H.todense(), eigvals_only=True)
				else:
					num = min(n_eigen, max(1, H.shape[0] // 2))

					if num >= H.shape[0]:
						eigvals = ln.eigh(H.todense(), eigvals_only=True)
					else:
						eigvals = sp.linalg.eigsh(
							H,
							k=num,
							ncv=max(2 * num + 1, 20),
							which='SA',
							return_eigenvectors=False)

				if print_time:
					total_time = time.time() - start_time
					print(f"\t\t• diagonalized Hamiltonian of in {total_time:.5f} s")

				sort = np.argsort(eigvals)
				energies[tuple(sorted(sector.items()))] = eigvals[sort]

	return energies

def diagonalize_exact(model, params, tweaks):
	"""
	Diagonalizes the Hamiltonian matrix exactly to find eigenvalues and eigenvectors.

	Args:
		model (dict): The model used to construct the Hamiltonian matrix.
		params (dict): Parameters for the Hamiltonian matrix.
		tweaks (dict): Additional tweaks for the diagonalization process.

	Returns:
		E (array): Eigenvalues of the Hamiltonian matrix.
		V (array): Eigenvectors of the Hamiltonian matrix.
	"""
	N = params.get('N', 0)
	n_eigen = tweaks.get('n_eigen', -1)

	H = Hamiltonian(model, params)

	e, v = ln.eigh(H.get_full_matrix(), eigvals_only=False)

	ind = np.argsort(e)

	E = e[ind]
	V = v[:, ind]

	if n_eigen is not None and n_eigen > 0:
		E = E[:n_eigen]
		V = V[:, :n_eigen]

	return E, V
