import os
import numpy as np
import pickle
import argparse

from tenet.linalg import SUIrrep
from tenet.algorithms import dmrg
from tenet.networks import mpo, mps
from tenet.utils import dict_to_str, DATA_FOLDER, PI

parser = argparse.ArgumentParser()
parser.add_argument("--lam", help="set variable lambda", type=float)
parser.add_argument("--theta", help="set variable theta", type=float)

args = parser.parse_args()

HPC = args.theta is not None or args.lam is not None

model_name = "A"
simul_name = "PS+PDC/BOTTOM"

lam = np.linspace(PI/4-PI/40, PI/4-PI/40, 1) if args.theta is not None and args.lam is None or not HPC else [args.lam]
theta = np.linspace(PI/4-PI/40, PI/4-PI/40, 1) if args.lam is not None and args.theta is None or not HPC else [args.theta]

N = np.linspace(10, 10, 1).astype(int)
chi = np.linspace(16, 32, 5).astype(int)

boundary_name = "NONE"
method_name = "IDMRG"

model = {"name": model_name}

lam, theta = np.round(lam, 15), np.round(theta, 15)

init_state = "neel"

d = 2

prec_var = 1e-6
prec_energy = 1e-6
prec_entropy = 1e-6
thresholds = {"variance": prec_var, "entropy": prec_entropy, "energy": prec_energy}

conv_variance = False
conv_energy = True
conv_entropy = True
convergence = {"variance": conv_variance, "energy": conv_energy, "entropy": conv_entropy}

prec_singular = 1e-9
prec_eigen = 1e-12
precision = {"svd": prec_singular, "eigen": prec_eigen}

num_sweeps_min = 1
num_sweeps_max = 1100
sweeps = {"max": num_sweeps_max, "min": num_sweeps_min}

k = 1
verbose = False
max_iter = 2
krylov_dim = 6
noise = 1e-9
noisy_sweeps = 3
options = {"k": k, "verbose": verbose, "max_iter": max_iter, "krylov_dim": krylov_dim, 'noise': noise, 'noisy_sweeps': noisy_sweeps}

save_every = 5
save = {'every': save_every}

base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

for l in lam:
	for t in theta:
		for n in N:
			for c in chi:
				folder = f"N={n}_chi={c}"

				params = {'lam': float(l), 'theta': float(t)}

				data_filename = f"N={n}_chi={c}_{dict_to_str(params)}.dat"

				filename = os.path.join(DATA_FOLDER, base_folder, folder, data_filename)

				save['file'] = filename

				if not os.path.exists(os.path.join(DATA_FOLDER, base_folder, folder, data_filename)):
					
					if not os.path.exists(os.path.join(DATA_FOLDER, base_folder, folder)):
						os.makedirs(os.path.join(DATA_FOLDER, base_folder, folder))

					H = mpo.iMPO(n, SUIrrep(2, d), model, params)
					psi = mps.iMPS(n, SUIrrep(2, d), init_state)
					
					engine = dmrg.iDMRGEngine(psi, H, c, sweeps, thresholds, convergence, precision, options, save)
					engine.run()

					with open(filename, 'wb') as file:
						pickle.dump(engine, file)
						
				else:

					# pass

					with open(filename, 'rb') as file:
						engine = pickle.load(file)

					print(c, l, t, engine.dE[-1], engine.dS[-1], engine.num_sweeps, engine.converged)

					# engine.num_sweeps_max = num_sweeps_max
					# engine.save_file = filename

					# engine._lighten_space()
					# engine.run()

				# with open(filename, 'wb') as file:
				# 	pickle.dump(engine, file)


