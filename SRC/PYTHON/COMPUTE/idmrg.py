import os
import numpy as np
import pickle
import scipy.sparse.linalg as ln

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
from tenet.utils.io import folder_names, load_file
from tenet.linalg.sun import SUIrrep
from tenet.networks.lin import TransferMatrix

su2 = SUIrrep(2, 2)
X, Y, Z = [2 * su2.generators[i] for i in range(3)]

force = False

with_vecs = False
compute_transfer_left = True
compute_transfer_right = False

compute_mags = True
compute_mags_spt = True
compute_corrs = True

model_name = "A"
simul_name = "PS+PDC/BOTTOM"

num_eigen = 10
r = 300

boundary_name = "NONE"
method_name = "IDMRG"

base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
		
	data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

	vals, data_filenames = load_file(data_folder, expression)

	N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

	for i in range(len(data_filenames)):
		
		print(data_filenames[i])

		quantities = {}

		if os.path.exists(os.path.join(data_folder, COMPUTE_FOLDER, data_filenames[i])):
			
			with open(os.path.join(data_folder, COMPUTE_FOLDER, data_filenames[i]), 'rb') as file:
				quantities = pickle.load(file)

			print("\t • ALREADY COMPUTED : ", [k for k in quantities.keys()])

		if force or not os.path.exists(os.path.join(data_folder, COMPUTE_FOLDER, data_filenames[i])):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)
			
			if compute_transfer_left:
				print("\t | transfer left...")
					
				k = num_eigen
				while k > 1:
					try:
						if with_vecs:
							quantities['val_L'], quantities['vec_L'] = ln.eigs(TransferMatrix(engine.psi, "L"), k=k, which='LM')
						else:
							quantities['val_L'] = ln.eigs(TransferMatrix(engine.psi, "L"), k=k, which='LM', return_eigenvectors=False)
					except:
						k = k - 10
						print(f"\t\t - Warning : num_eigen reduced to {k}")
					else:
						k = 0

			if compute_transfer_right:
				print("\t | transfer right...")
					
				k = num_eigen
				while k > 1:
					try:
						if with_vecs:
							quantities['val_R'], quantities['vec_R'] = ln.eigs(TransferMatrix(engine.psi, "R"), k=k, which='LM')
						else:
							quantities['val_R'] = ln.eigs(TransferMatrix(engine.psi, "R"), k=k, which='LM', return_eigenvectors=False)
					except:
						k = k - 10
						print(f"\t\t - Warning : num_eigen reduced to {k}")
					else:
						k = 0

			if compute_mags:
				print("\t | mags...")

				spin_X, spin_Y, spin_Z = engine.psi.compute_spin()

				quantities['mags'] = np.array([spin_X, spin_Y, spin_Z])

			if compute_mags_spt:
				print("\t | mags SPT...")

				spins_spt = np.zeros((3, N-2))

				spins_spt[0, :] = [engine.psi.compute_ev_3([Z, X, Z], l) for l in range(N-2)]
				spins_spt[1, :] = [engine.psi.compute_ev_3([Z, Y, Z], l) for l in range(N-2)]
				spins_spt[2, :] = [engine.psi.compute_ev_3([Z, Z, Z], l) for l in range(N-2)]

				quantities['mags_spt'] = spins_spt

			if compute_corrs:
				print("\t | corrs...")

				corr_X = engine.psi.compute_correlation([X, X], r)
				corr_Y = engine.psi.compute_correlation([Y, Y], r)
				corr_Z = engine.psi.compute_correlation([Z, Z], r)

				quantities['corrs'] = np.array([corr_X, corr_Y, corr_Z])

			if not os.path.exists(os.path.join(data_folder, COMPUTE_FOLDER)):
				os.makedirs(os.path.join(data_folder, COMPUTE_FOLDER))

			with open(os.path.join(data_folder, COMPUTE_FOLDER, data_filenames[i]), 'wb') as file:
				pickle.dump(quantities, file)