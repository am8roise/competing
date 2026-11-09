__all__ = [
	'energies_aggregated',
	'energies_aggregated_with_keys',
	'energies_degeneracies',
	'specific_heat_from_energies',
]

import numpy as np

def energies_aggregated(energies, n):
	"""
	Sorts energy values by sector type.

	Args:
		energies (dict): A dictionary containing energy values for different sectors where keys are tuples like tuple({'M': m, 'K': k}.items()).
		n (int): Number of top energy values to be returned for each sector.

	Returns:
		tuple: A tuple containing two dictionaries, EM and EK, where energy values are sorted by sector type 'M' and 'K' respectively.
	"""
	EM = {}
	EK = {}
	E = []

	for key, val in energies.items():
		E.extend(val)

		sector = dict(key)
		M = sector.get('M', None)
		K = sector.get('K', None)

		if M is not None:
			if M not in EM:
				EM[M] = []
			EM[M].extend(val)

		if K is not None:
			if K not in EK:
				EK[K] = []
			EK[K].extend(val)

	for M in EM:
		EM[M] = sorted(EM[M])[:n]
	for K in EK:
		EK[K] = sorted(EK[K])[:n]

	return EM, EK, sorted(E)[:n]

def energies_aggregated_with_keys(energies, n):
	"""
	Aggregate energy values based on keys 'K' and 'M' and return the top 'n' values for each key.

	Args:
		energies (dict): A dictionary containing energy values with keys as sectors.
		n (int): Number of top energy values to be returned for each key.

	Returns:
		tuple: Two dictionaries containing aggregated energy values for keys 'M' and 'K' respectively.
	"""
	EM = {}
	EK = {}
	E = []

	for key, val in energies.items():
		E.extend(val)
				
		sector = dict(key)
		K = sector.get('K', None)
		M = sector.get('M', None)

		if M is not None and K is not None:
			if M not in EM:
				EM[M] = []
			if K not in EK:
				EK[K] = []

			for energy in val:
				EM[M].append((energy, K))
				EK[K].append((energy, M))

	for M in EM:
		EM[M] = sorted(EM[M], key=lambda x: x[0])[:n]
	for K in EK:
		EK[K] = sorted(EK[K], key=lambda x: x[0])[:n]

	return EM, EK, sorted(E)[:n]

def energies_degeneracies(E, n, tol=1e-10):
	"""
	Compute degeneracies of the lowest `n` unique energy levels across all magnetization/momentum sectors.

	Parameters:
		E (dict): Maps sectors S → list of sorted energies.
		n (int): Number of lowest unique energies to analyze.
		tol (float): Tolerance for comparing floating-point energies.

	Returns:
		List of tuples: (energy, degeneracy, list of magnetization/momentum sectors contributing)
	"""
	all_energies = []

	for key, energies in E.items():
		for e in energies:
			all_energies.append((e, key))

	all_energies.sort(key=lambda x: x[0])

	grouped = []
	current_group = [all_energies[0]]

	for e, key in all_energies[1:]:
		if abs(e - current_group[0][0]) < tol:
			current_group.append((e, key))
		else:
			grouped.append(current_group)
			current_group = [(e, key)]
	grouped.append(current_group)

	result = []
	for group in grouped[:n]:
		energy_val = group[0][0]
		deg = len(group)
		sectors = sorted([key for (_, key) in group])
		result.append((energy_val, deg, sectors))

	return result

def specific_heat_from_energies(E, T, N=None):
	"""
	Compute specific heat C(T) from a list/array of ordered energies E.
	
	Parameters
	----------
	energies : array-like
		Energy levels [E0, E1, E2, ...] for a given finite system size.
	T : float
		Temperature (k_B = 1). Must be > 0.
	L : int or None
		System size (number of sites). If None, returns C(T) not normalized per site.

	Returns
	-------
	C : float
		Specific heat at temperature T. If L is provided, returns C per site.
	"""
	E = np.asarray(E, dtype=np.float64)

	if E.ndim != 1 or E.size == 0 or not np.isfinite(E).all():
		raise ValueError("`E` must be a 1D array of finite numbers with at least one element.")

	if T <= 0:
		raise ValueError("T must be positive.")


	beta = 1.0 / T
	dE = E - E[0]

	w = np.exp(-beta * dE)
	Z = np.sum(w)

	ev_E1 = np.sum(dE * w) / Z
	ev_E2 = np.sum((dE ** 2) * w) / Z

	var_E = ev_E2 - ev_E1 ** 2

	C = var_E / (T ** 2)
	if N is not None and N > 0:
		C = C / N

	return C