import os
import time
import numpy as np
import scipy.sparse as sp
import pickle
import argparse

from tenet.algorithms.ed import diagonalize, diagonalize_exact, print_parity_gap_diagnostic
from tenet.utils.path import DATA_FOLDER
from tenet.utils.io import dict_to_str
from tenet.utils.math import PI

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the result to file.")
parser.add_argument("--print", action="store_true", help="Print the results in the terminal.")
parser.add_argument("--lam", help="Set variable lambda.", type=float)
parser.add_argument("--theta", help="Set variable theta.", type=float)

args = parser.parse_args()

HPC = args.theta is not None or args.lam is not None

args = parser.parse_args()

def main():
	"""
	Executes the diagonalization process.
	"""
	exact = True

	model_name = "B"
	simul_name = "TEST/BOTTOM"

	lam = np.linspace(3*PI/4, 3*PI/4, 1) if args.theta is not None and args.lam is None or not HPC else [args.lam]
	theta = np.linspace(PI/2+0.01, PI/2+0.05, 1) if args.lam is not None and args.theta is None or not HPC else [args.theta]

	N = np.linspace(8, 10, 1).astype(int)

	n_eigen = 10
	print_time = True

	PBCs = True

	magnetization = True
	parity = True
	momentum = True if PBCs else False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	model = {'name': model_name, 'BCs': boundary_name}
	tweaks = {'n_eigen': n_eigen, 'print_time': print_time}
	blocks = {'M': magnetization, 'K': momentum, 'P': parity}

	lam, theta = np.round(lam, 15), np.round(theta, 15)

	for l in lam:
		for t in theta:
			for n in N:
				
				params = {'N': n, 'lam': float(l), 'theta': float(t)}

				folder = f"N={params['N']}"
				data_folder = os.path.join(DATA_FOLDER, base_folder, folder)
				data_filename = os.path.join(data_folder, f"{dict_to_str(params)}.dat")

				print(f"\n\t\t start ED • {model['name']} • {model['BCs']}")
				print(f"\n\t | 2N : {params['N']}")
				print(f"\t | λ : {params['lam']}")
				print(f"\t | θ : {params['theta']}")

				if not os.path.exists(data_filename):

					if args.save and not os.path.exists(data_folder):
						os.makedirs(data_folder)

					if exact:
						start_time = time.time()
						vals, vecs = diagonalize_exact(model, params, tweaks)

						engine = {"energies": vals, "eigenvectors": vecs}
						total_time = time.time() - start_time

					else:
						start_time = time.time()
						energies = diagonalize(model, params, tweaks, blocks)
						total_time = time.time() - start_time

						engine = {"energies": energies}

					if args.save:
						with open(data_filename, 'wb') as file:
							pickle.dump(engine, file)

					if args.print:
						if exact:
							print("\nE =", engine["energies"])
						else:
							print("\nE =", np.sort(np.concatenate(list(engine["energies"].values())))[:n_eigen])

					print("\n", "-" * 50, " DONE in : ", f"{total_time:.5f} s", sep="")
				else:
					print("\n", "-" * 50, " ALREADY DONE", sep="")

if __name__ == "__main__":
	main()
