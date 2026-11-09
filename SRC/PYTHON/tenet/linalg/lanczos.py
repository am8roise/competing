import numpy as np
from numpy.linalg import norm, eigh

__all__ = [
	'restarted_lanczos'
]

def lanczos_iteration(H, v, m):
	"""
	Perform m steps of Lanczos iteration starting from v.
	Returns: alpha, beta, Q (orthonormal basis)
	"""
	n = len(v)
	Q = np.zeros((n, m), dtype=complex)
	alpha = np.zeros(m)
	beta = np.zeros(m - 1)

	q = v / norm(v)
	Q[:, 0] = q
	w = H._matvec(q)
	alpha[0] = np.vdot(q, w).real
	w = w - alpha[0] * q

	for j in range(1, m):
		beta[j - 1] = norm(w)
		if beta[j - 1] < 1e-14:
			Q = Q[:, :j]
			alpha = alpha[:j]
			beta = beta[:j - 1]
			break

		q = w / beta[j - 1]
		Q[:, j] = q
		w = H._matvec(q)
		w = w - beta[j - 1] * Q[:, j - 1]
		alpha[j] = np.vdot(q, w).real
		w = w - alpha[j] * q

	return alpha, beta, Q

def restarted_lanczos(H, v, m, k, max_restarts=10, tol=1e-10):
	"""
	Restarted Lanczos using `lanczos_iteration`.
	Computes k lowest eigenpairs of H.
	"""
	n = len(v)
	assert m > k

	q = v / norm(v)

	for r in range(max_restarts):
		alpha, beta, Q = lanczos_iteration(H, q, m)

		# Tridiagonal matrix T
		m_eff = len(alpha)
		T = np.diag(alpha)
		if m_eff > 1:
			T += np.diag(beta[:m_eff - 1], 1) + np.diag(beta[:m_eff - 1], -1)

		# Ritz values and vectors
		evals_T, evecs_T = eigh(T)
		idx = np.argsort(evals_T)
		evals = evals_T[idx[:k]]
		ritz_vecs = Q @ evecs_T[:, idx[:k]]

		# Check residuals
		residuals = [norm(H._matvec(ritz_vecs[:, n]) - evals[n] * ritz_vecs[:, n]) for n in range(ritz_vecs.shape[1])]
		if max(residuals) < tol:
			return evals, ritz_vecs

		# Restart with best Ritz vector
		q = ritz_vecs[:, 0]

	return evals, ritz_vecs