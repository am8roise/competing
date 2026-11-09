import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as sco
import argparse
from cycler import cycler
import os

from tenet.utils.io import save_graph
from tenet.utils.sty import COLORS

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

gapless = False

if gapless:

	def eps_n(k, n, chi, lam):
		return (4*lam*n*(1 - n) - 1) * 2*np.cos(k) - 4*lam*(2*n - 1)*chi

	def fd_n(k, n, chi, lam):
		return np.heaviside(-eps_n(k, n, chi, lam), 1) 
		
	def loop(roots, params):

		lam = params[0]
		k = params[1]
		N = params[2]

		n = roots[0]
		chi = roots[1]

		n_next =  np.sum(fd_n(k, n, chi, lam)) / (2*N)
		chi_next = np.sum(2*np.cos(k) * (fd_n(k, n, chi, lam))) / (2*N)

		return [n - n_next, chi - chi_next]

	n_lam = 200
	lam = np.linspace(0, 0.999, n_lam)

	n_0 = [1, 0, 0.5]
	chi_0 = [1, 1, 1]

	n_0_labels = ["1", "0", r"\tfrac{1}{2}"]

	n_init = len(n_0)
	assert n_init == len(chi_0)

	N = 10000

	k = np.zeros(2*N)
	for x in range(-N, N):
		k[x] = (2*np.pi / (2*N)) * x

	n = np.zeros((n_lam, n_init))
	chi = np.zeros((n_lam, n_init))

	for i in range(n_init):
		
		for l in range(n_lam):

			sol = sco.root(loop, [n_0[i], chi_0[i]], args=[lam[l], k, N], method='hybr', jac=None, options={'xtol': 1e-13})
			n[l, i], chi[l, i] = sol.x

	cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[0],COLORS[2],COLORS[3]], N=256)

	plt.rcParams.update({'axes.prop_cycle': cycler(linestyle=(["-", (0, (2, 4)), (0, (0, 2))]), color=cmap(np.linspace(0, 1, n_init)))})

	fig, ax = plt.subplots(2, 1, gridspec_kw={"hspace": 0.2, "height_ratios": [1.3, 1]}, figsize=(2.3, 3))

	handles = []
	for i in range(n_init):
		p, = ax[0].plot(lam, n[:, i], lw=2, label=fr"$\bigl(\bar n^{{\,}}_{{0}},\bar{{\chi}}^{{\,}}_{{0}}\bigr)=\bigl({n_0_labels[i]},1\bigr)$")
		handles.append(p)

	for i in range(n_init):
		p, = ax[1].plot(lam, chi[:, i], lw=2)

	ax[0].legend(
		handles=handles,
		loc="lower left",
		ncol=1,
		labelspacing=0.4,
		handlelength=1.2,
		handletextpad=0.5,
		bbox_to_anchor=(0.05, 0.9)).set_zorder(1000)	

	ax[0].set_xlim(-0.05, 1.05)
	ax[1].set_xlim(-0.05, 1.05)

	ax[0].set_xticks([])
	ax[1].set_xticks([0, 0.5, 1])
	ax[1].set_xticklabels(["$0$", r"$\tfrac{1}{2}$", "$1$"])

	ax[0].set_yticks([0, 0.5, 1])
	ax[0].set_yticklabels(["$0$", r"$\tfrac{1}{2}$", "$1$"])
	ax[1].set_yticks([1/np.pi, 2/np.pi])
	ax[1].set_yticklabels([r"$\tfrac{1}{\pi}$", r"$\tfrac{2}{\pi}$"])

	ax[1].set_xlabel(r"$\epsilon$")

	ax[0].set_ylabel(r"$\bar{n}$")
	ax[1].set_ylabel(r"$\bar{\chi}$")

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join("./"), f"gapless")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

else:

	n_lam = 200
	lam = np.linspace(0, 0.99, n_lam)

	n_0 = [-1, 1, 0]
	chi_0 = [1, 1, 1]

	n_0_labels = ["-1", "1", "0"]

	n_init = len(n_0)
	assert n_init == len(chi_0)

	N = 10000

	k = np.zeros(N)
	for x in range(-N, 0):
		k[x] = (2*np.pi / (2*N)) * x

	n = np.zeros((n_lam, n_init))
	chi = np.zeros((n_lam, n_init))

	def t_n(k, n, chi, lam):
		return (1 - lam * (1 - 4 * n ** 2)) * 2 * np.cos(k)

	def mu_n(k, n, chi, lam):
		return 8 * lam * chi * n

	def eps_n(k, n, chi, lam):
		return -np.sqrt(t_n(k, n, chi, lam) ** 2 + mu_n(k, n, chi, lam) ** 2)

	def fd_n(k, n, chi, lam):
		return np.heaviside(-eps_n(k, n, chi, lam), 1) 
		
	def loop(roots, lam):

		n = roots[0]
		chi = roots[1]

		n_next =  np.sum(fd_n(k, n, chi, lam) * mu_n(k, n, chi, lam) / np.abs(eps_n(k, n, chi, lam))) / (2 * N)
		chi_next = np.sum(2 * np.cos(k) * fd_n(k, n, chi, lam) * t_n(k, n, chi, lam) / np.abs(eps_n(k, n, chi, lam))) / (2 * N)

		return [n - n_next, chi - chi_next]

	for i in range(n_init):
		
		for l in range(n_lam):	

			sol = sco.root(loop, [n_0[i], chi_0[i]], args=lam[l], method='hybr', jac=None, options={'xtol': 1e-13})
			n[l, i], chi[l, i] = sol.x

	cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[0],COLORS[2],COLORS[3]], N=256)

	plt.rcParams.update({'axes.prop_cycle': cycler(linestyle=(["-", (0, (2, 4)), (0, (0, 2))]), color=cmap(np.linspace(0, 1, n_init)))})

	fig, ax = plt.subplots(2, 1, gridspec_kw={"hspace": 0.2, "height_ratios": [1.3, 1]}, figsize=(2.3, 3))

	handles = []
	for i in range(n_init):
		p, = ax[0].plot(lam, n[:, i], lw=2, label=fr"$\bigl(\bar n^{{\,}}_{{0}},\bar{{\chi}}^{{\,}}_{{0}}\bigr)=\bigl({n_0_labels[i]},1\bigr)$")
		handles.append(p)

	for i in range(n_init):
		p, = ax[1].plot(lam, chi[:, i], lw=2)

	ax[0].legend(
		handles=handles,
		loc="lower right",
		ncol=1,
		labelspacing=0.4,
		handlelength=1.2,
		handletextpad=0.5,
		bbox_to_anchor=(1.05, 0.9)).set_zorder(1000)	

	ax[0].set_xlim(-0.05, 1.05)
	ax[1].set_xlim(-0.05, 1.05)

	ax[0].set_xticks([])
	ax[1].set_xticks([0, 0.5, 1])
	ax[1].set_xticklabels(["$0$", r"$\tfrac{1}{2}$", "$1$"])

	ax[0].set_yticks([-0.5, 0, 0.5])
	ax[0].set_yticklabels([r"$-\tfrac{1}{2}$", "$0$", r"$\tfrac{1}{2}$"])
	ax[1].set_yticks([1/np.pi, 2/np.pi])
	ax[1].set_yticklabels([r"$\tfrac{1}{\pi}$", r"$\tfrac{2}{\pi}$"])

	ax[1].set_xlabel(r"$\epsilon$")

	ax[0].set_ylabel(r"$\bar{n}$", labelpad=-5)
	ax[1].set_ylabel(r"$\bar{\chi}$")

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join("./"), f"gapped")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

