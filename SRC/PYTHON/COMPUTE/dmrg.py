import os
import numpy as np
import pickle

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
from tenet.utils.io import folder_names, load_file
from tenet.linalg.sun import SUIrrep

su2 = SUIrrep(2, 4)
X, Y, Z = [2 * su2.generators[i] for i in range(3)]

PBCs = False
force = False

compute_entropy = True
compute_mags = True
compute_mags_spt = False
compute_corrs = False
compute_evs_2 = False
compute_evs_4 = False
compute_evs_6 = False
compute_evs_8 = False
compute_evs_10 = False

model_name = "B"
simul_name = "4_PDC/BOTTOM"

r = 64
s = 16

boundary_name = "PBC" if PBCs else "OBC"
method_name = "DMRG"

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
				
			if compute_entropy:
				print("\t | entropy...")

				quantities['entropy'] = engine.psi.compute_entropy()

			if compute_mags:
				print("\t | mags...")

				quantities["mags"] = np.array(engine.psi.compute_spin())

			if compute_mags_spt:
				print("\t | magsSPT...")

				# num_sites = engine.N if PBCs else engine.N - 2
				num_sites = engine.N if PBCs else engine.N - 3
				magsSPT = np.zeros((3, num_sites))

				# magsSPT[0, :] = [engine.psi.compute_ev_3([Z, X, Z], j) for j in range(num_sites)]
				# magsSPT[1, :] = [engine.psi.compute_ev_3([Z, Y, Z], j) for j in range(num_sites)]
				# magsSPT[2, :] = [engine.psi.compute_ev_3([Z, Z, Z], j) for j in range(num_sites)]

				magsSPT[0, :] = [engine.psi.compute_ev_4([Z, X, X, Z], j) for j in range(num_sites)]
				magsSPT[1, :] = [engine.psi.compute_ev_4([Z, Y, Y, Z], j) for j in range(num_sites)]
				magsSPT[2, :] = [engine.psi.compute_ev_4([Z, Z, Z, Z], j) for j in range(num_sites)]

				quantities["mags_spt"] = magsSPT

			if compute_corrs:
				print("\t | corrs...")

				start = engine.N // 2 - s // 2

				corr_X = engine.psi.compute_correlation([X, X], start, s)
				corr_Y = engine.psi.compute_correlation([Y, Y], start, s)
				corr_Z = engine.psi.compute_correlation([Z, Z], start, s)

				quantities['corrs'] = np.array([corr_X, corr_Y, corr_Z])

			if compute_evs_2:
				print("\t | evs_2...")

				evs_2 = np.zeros((3, r))
				
				evs_2x = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_2x.append(engine.psi.compute_ev_2([X, X], j))
				evs_2[0, :] = evs_2x

				evs_2y = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_2y.append(engine.psi.compute_ev_2([Y, Y], j))
				evs_2[1, :] = evs_2y
				
				evs_2z = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_2z.append(engine.psi.compute_ev_2([Z, Z], j))
				evs_2[2, :] = evs_2z

				quantities["evs_2"] = evs_2

			if compute_evs_4:
				print("\t | evs_4...")

				evs_4 = np.zeros((3, r))
				
				evs_4x = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_4x.append(engine.psi.compute_ev_4([Z, X, X, Z], j))
				evs_4[0, :] = evs_4x

				evs_4y = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_4y.append(engine.psi.compute_ev_4([Z, Y, Y, Z], j))
				evs_4[1, :] = evs_4y
				
				evs_4z = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_4z.append(engine.psi.compute_ev_4([Z, Z, Z, Z], j))
				evs_4[2, :] = evs_4z

				quantities["evs_4"] = evs_4

			if compute_evs_6:
				print("\t | evs_6...")

				evs_6 = np.zeros((3, r))
				
				evs_6x = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_6x.append(engine.psi.compute_ev_6([Z, X, X, X, X, Z], j))
				evs_6[0, :] = evs_6x

				evs_6y = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_6y.append(engine.psi.compute_ev_6([Z, Y, Y, Y, Y, Z], j))
				evs_6[1, :] = evs_6y
				
				evs_6z = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_6z.append(engine.psi.compute_ev_6([Z, Z, Z, Z, Z, Z], j))
				evs_6[2, :] = evs_6z

				quantities["evs_6"] = evs_6

			if compute_evs_8:
				print("\t | evs_8...")

				evs_8 = np.zeros((3, r))
				
				evs_8x = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_8x.append(engine.psi.compute_ev_8([Z, X, X, X, X, X, X, Z], j))
				evs_8[0, :] = evs_8x

				evs_8y = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_8y.append(engine.psi.compute_ev_8([Z, Y, Y, Y, Y, Y, Y, Z], j))
				evs_8[1, :] = evs_8y
				
				evs_8z = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_8z.append(engine.psi.compute_ev_8([Z, Z, Z, Z, Z, Z, Z, Z], j))
				evs_8[2, :] = evs_8z

				quantities["evs_8"] = evs_8

			if compute_evs_10:
				print("\t | evs_10...")

				evs_10 = np.zeros((3, r))
				
				evs_10x = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_10x.append(engine.psi.compute_ev_10([Z, X, X, X, X, X, X, X, X, Z], j))
				evs_10[0, :] = evs_10x

				evs_10y = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_10y.append(engine.psi.compute_ev_10([Z, Y, Y, Y, Y, Y, Y, Y, Y, Z], j))
				evs_10[1, :] = evs_10y
				
				evs_10z = []
				for j in range(engine.N // 2 - r // 2, engine.N // 2 + r // 2):
					evs_10z.append(engine.psi.compute_ev_10([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z], j))
				evs_10[2, :] = evs_10z

				quantities["evs_10"] = evs_10

			if not os.path.exists(os.path.join(data_folder, COMPUTE_FOLDER)):
				os.makedirs(os.path.join(data_folder, COMPUTE_FOLDER))

			with open(os.path.join(data_folder, COMPUTE_FOLDER, data_filenames[i]), 'wb') as file:
				pickle.dump(quantities, file)