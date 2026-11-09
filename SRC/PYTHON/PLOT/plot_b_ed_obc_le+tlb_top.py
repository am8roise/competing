import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from collections import defaultdict
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.sty import COLORS
from tenet.utils.io import folder_names, load_file, save_graph, format_value, format_sign
from tenet.utils.helpers import energies_aggregated
from tenet.utils.math import r2

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "LE+TLB/TOP"

	PBCs = False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	num = 8
	
	ns = [0, 1, 3, 5]
	ns_1 = [0, 3, 5]
	ns_2 = [1]
	n_fit = 20

	Ns = []
	diffs_Ns = []
	
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		diffs = np.zeros((len(lam), len(theta), num - 1))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			E = engine["energies"]

			EM, EK, E = energies_aggregated(E, num)
			E = np.array(E)

			diffs[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = E[1:] - E[:-1]
					
		Ns.append(N)
		diffs_Ns.append(diffs)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

	As = np.zeros((len(lam), len(theta), num - 1, 2))
	Bs = np.zeros((len(lam), len(theta), num - 1, 2))

	for t in range(len(theta)):
		for l in range(len(lam)):

			for n in ns:

				def f(N, a, b):
					return a * np.log(N) - 3 * np.log(N-1) + b

				def g(N, a, b):
					return a * 1/N + b

				ind = np.where(Ns >= 20)[0]

				if n == 0:
					popt, pcov = sco.curve_fit(f, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))
		
					perr = np.sqrt(np.diag(pcov))

					a = np.exp(popt[1])
					da = perr[1] * np.exp(popt[1])
				else:
					popt, pcov = sco.curve_fit(g, Ns[ind], diffs_Ns[ind, l, t, n])
						
					perr = np.sqrt(np.diag(pcov))

					a = popt[0]
					da = perr[0]

				As[lam[l]==lam, theta[t]==theta, n, :] = [a, da]

	linestyles = ["-", "-", "", ":", "", ":"]
	markers = ["x", fr"$\setminus$", "", ".", "", "$/$"]
	markersizes = [5, 8, 0, 10, 0, 8]
	colors = [COLORS[0],COLORS[2],"",COLORS[3],"",COLORS[5]]

	fig, ax = plt.subplots(figsize=(8, 3))

	handles = []
	s = 0

	for n in ns:
		for t in range(len(theta)):
				
			l = 20
			print(lam[l])
		
			def f(N, a, b):
				return a * np.log(N) - 3 * np.log(N-1) + b

			def g(N, a, b):
				return a * np.log(N) + b

			ind = np.where(Ns >= 8)[0]

			p, = ax.plot(np.log(Ns[ind]), np.log(diffs_Ns[ind, l, t, n]), color=colors[n], marker=markers[n], markersize=markersizes[n], linewidth=0, markeredgewidth=2, zorder=99, label=fr"$E^{{\,}}_{{{n+1},{n}}}$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
			handles.insert(s, p)

			ind = np.where(Ns >= 20)[0]

			if n == 0:
				popt, pcov = sco.curve_fit(f, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))
			else:
				popt, pcov = sco.curve_fit(g, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))
			perr = np.sqrt(np.diag(pcov))

			a = np.exp(popt[1])
			da = perr[1] * np.exp(popt[1])

			if n == 0:
				R2 = r2(np.log(diffs_Ns[ind, l, t, n]), f(Ns[ind], *popt))

				o, = ax.plot(np.log(Ns[ind]), f(Ns[ind], *popt), color=p.get_color(), linestyle=linestyles[n], linewidth=1, markersize=0, zorder=98, label=fr"$({format_value(a, da)})\times (2N)^{{{format_value(popt[0], perr[0])}}} \times (2N-1)^{{-3}}$")
				o.set_marker(mpl.markers.MarkerStyle(o.get_marker(), capstyle="round"))
			else:
				R2 = r2(np.log(diffs_Ns[ind, l, t, n]), g(Ns[ind], *popt))

				o, = ax.plot(np.log(Ns[ind]), g(Ns[ind], *popt), color=p.get_color(), linestyle=linestyles[n], linewidth=1, markersize=0, zorder=98, label=fr"$({format_value(a, da)})\times (2N)^{{{format_value(popt[0], perr[0])}}}$")
				o.set_marker(mpl.markers.MarkerStyle(o.get_marker(), capstyle="round"))

			handles.append(o)
			s += 1

	ax.minorticks_on()

	plt.xlabel(r"$\ln\,2N$")

	ax.legend(handles=handles, loc="lower center", ncol=2, labelspacing=0.6, columnspacing=1, handletextpad=0.5, handlelength=1.4, borderaxespad=0, bbox_to_anchor=(0.5, 1.03)).set_zorder(200)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"alg")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

	linestyles = ["-", "-", "", ":", "", ":"]
	markers = ["x", fr"$\setminus$", "", ".", "", "$/$"]
	markersizes = [5, 8, 0, 10, 0, 8]
	colors = [COLORS[0],COLORS[2],"",COLORS[3],"",COLORS[5]]

	fig, ax = plt.subplots(figsize=(8, 3))

	axin = ax.inset_axes([0.12, 0.35, 0.4, 0.35])
	axin.set_zorder(201)

	handles = []
	s = 0

	for n in ns:
		for t in range(len(theta)):

			def g(x, a, b, c):
				return a * x ** b + c

			if n == 0:
				label = fr"$v^{{\prime}}_{{{n+1},{n}}}$"
			else:
				label = fr"$v^{{\,}}_{{{n+1},{n}}}$"

			if n in ns_1:

				p, = ax.plot(lam - np.pi/2, As[:, t, n, 0], linestyle="none", marker=markers[n], color=colors[n], markersize=markersizes[n], markeredgewidth=2, zorder=99-n, label=label)
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				handles.insert(s, p)

				popt, pcov = sco.curve_fit(g, lam[:n_fit] - np.pi/2, As[:n_fit, t, n, 0], [10, 2, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				x = np.linspace(0, 0.05, 100)

				label = fr"$({format_value(popt[0], perr[0])}) \cdot (\lambda-\pi/2)^{{{format_value(popt[1], perr[1])}}} {format_sign(popt[2])} {format_value(np.abs(popt[2]), perr[2])}$"
				p, = ax.plot(x,  g(x, *popt), color=colors[n], linestyle=linestyles[n], linewidth=1.5, markersize=0, zorder=99-n, label=label)
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				handles.append(p)

			elif n in ns_2:

				p, = axin.plot(lam - np.pi/2, As[:, t, n, 0], linestyle="none", marker=markers[n], color=colors[n], markersize=markersizes[n], markeredgewidth=2, zorder=99-n, label=fr"$v^{{\,}}_{{{n+1},{n}}}$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				handles.insert(s, p)

				popt, pcov = sco.curve_fit(g, lam[:n_fit] - np.pi/2, As[:n_fit, t, n, 0], [10, 2, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				x = np.linspace(0, 0.05, 100)

				label = fr"$({format_value(popt[0], perr[0])}) \cdot (\lambda-\pi/2)^{{{format_value(popt[1], perr[1])}}} {format_sign(popt[2])} {format_value(np.abs(popt[2]), perr[2])}$"
				p, = axin.plot(x,  g(x, *popt), color=colors[n], linestyle=linestyles[n], linewidth=1.5, markersize=0, zorder=99-n, label=label)
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				handles.append(p)

			s += 1

	ax.minorticks_on()
	axin.minorticks_on()

	ax.set_xlim(0, 0.05)
	axin.set_xlim(0, 0.05)

	axin.set_ylim(-0.01, 0.05)

	ax.set_xlabel(r"$\lambda-\pi/2$")

	ax.legend(handles=handles, loc="lower center", ncol=2, labelspacing=0.6, columnspacing=1, handletextpad=0.5, handlelength=1.4, borderaxespad=0, bbox_to_anchor=(0.5, 1.03)).set_zorder(200)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"velocities")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()
	
if __name__ == "__main__":
	main()
