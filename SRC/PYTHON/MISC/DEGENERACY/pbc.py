import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os

from tenet.utils import PI
from tenet.utils.io import save_graph
from tenet.utils.sty import COLORS

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def count_zero_modes(S, tol):

	return len(np.where(S < tol)[0])

c1 = 1
c2 = 3

Ns = np.arange(6, 26+1, 2)

for N in Ns:

	# lam = pi / 4

	k_cont = 2*PI/N * np.linspace(-N//2, N//2, 1000)
	k_e = 2*PI/N * np.arange(-N//2+1, N//2+1)
	k_o = 2*PI/N * np.arange(-N//2+1, N//2+1) - PI/N

	norm = 2 * np.sqrt(2)

	eps_cont = np.abs(np.cos(k_cont * (c1+c2)/2)) * norm
	eps_e = np.abs(np.cos(k_e * (c1+c2)/2)) * norm
	eps_o = np.abs(np.cos(k_o * (c1+c2)/2)) * norm
	
	print(" • lam : pi / 4")
	print(" • size : ", N)
	print(" • degeneracy (even) : ", 2 ** count_zero_modes(eps_e, 1E-10))
	print(" • degeneracy (odd) : ", 2 ** count_zero_modes(eps_o, 1E-10))
	print(" • energy (even) : ", -np.sum(eps_e))
	print(" • energy (odd) : ", -np.sum(eps_o))

	print(" ··· 2/tan(2 pi/N) = ", 2/np.tan(2*PI/N) * norm)
	print(" ··· 2/sin(2 pi/N) = ", 2/np.sin(2*PI/N) * norm)
	print(" ··· 1/sin(pi/N) = ", 1/np.sin(PI/N) * norm)

	fig, ax = plt.subplots(figsize=(5, 4))

	ax.plot(k_cont, eps_cont, marker="none", linestyle="--", color=COLORS[4], linewidth=1, label=r"$\varepsilon^{+}_{k}$")
	ax.plot(k_cont, -eps_cont, marker="none", color=COLORS[4], linewidth=1, label=r"$\varepsilon^{-}_{k}$")

	p, = ax.plot(k_e, -eps_e, marker="o", linewidth=0, color=COLORS[2], label=r"$\varepsilon^{-}_{k^{\,}_{\mathrm{even}}}$")
	ax.plot(k_o, -eps_o, marker="o", linewidth=0, color=COLORS[0], label=r"$\varepsilon^{-}_{k^{\,}_{\mathrm{odd}}}$")
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	ax.minorticks_on()

	ax.set_xlabel(r"$k$")

	ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, handlelength=1.5, ncol=1, labelspacing=0.8).set_zorder(200)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join("./"), f"pbc_lam={PI/4}_N={N}")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

	# lam = 3 pi / 4

	k_cont = 2*PI/N * np.linspace(-N//2, N//2, 1000)
	k_e = 2*PI/N * np.arange(-N//2+1, N//2+1)
	k_o = 2*PI/N * np.arange(-N//2+1, N//2+1) - PI/N

	norm = 2 * np.sqrt(2)

	eps_cont = np.abs(np.sin(k_cont * (c1+c2)/2)) * norm
	eps_e = np.abs(np.sin(k_e * (c1+c2)/2)) * norm
	eps_o = np.abs(np.sin(k_o * (c1+c2)/2)) * norm
	
	print(" • lam : 3 pi / 4")
	print(" • size : ", N)
	print(" • degeneracy (even) : ", 2 ** count_zero_modes(eps_e, 1E-10))
	print(" • degeneracy (odd) : ", 2 ** count_zero_modes(eps_o, 1E-10))
	print(" • energy (even) : ", -np.sum(eps_e))
	print(" • energy (odd) : ", -np.sum(eps_o))

	print(" ··· 2/tan(2 pi/N) = ", 2/np.tan(2*PI/N) * norm)
	print(" ··· 2/sin(2 pi/N) = ", 2/np.sin(2*PI/N) * norm)
	print(" ··· 1/tan(pi/N) = ", 1/np.tan(PI/N) * norm)

	fig, ax = plt.subplots(figsize=(5, 4))

	ax.plot(k_cont, eps_cont, marker="none", linestyle="--", color=COLORS[4], linewidth=1, label=r"$\varepsilon^{+}_{k}$")
	ax.plot(k_cont, -eps_cont, marker="none", color=COLORS[4], linewidth=1, label=r"$\varepsilon^{-}_{k}$")

	p, = ax.plot(k_e, -eps_e, marker="o", linewidth=0, color=COLORS[2], label=r"$\varepsilon^{-}_{k^{\,}_{\mathrm{even}}}$")
	ax.plot(k_o, -eps_o, marker="o", linewidth=0, color=COLORS[0], label=r"$\varepsilon^{-}_{k^{\,}_{\mathrm{odd}}}$")
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	ax.minorticks_on()

	ax.set_xlabel(r"$k$")

	ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, handlelength=1.5, ncol=1, labelspacing=0.8).set_zorder(200)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join("./"), f"pbc_lam={3*PI/4}_N={N}")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()